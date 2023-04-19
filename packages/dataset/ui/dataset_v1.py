"""Dataset initialization (v1). Configures the dataset for an experiment."""

import argparse
from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from gna.env import env
from itertools import chain
from gna.dataset import Dataset
from gna.parameters.parameter_loader import get_parameters, get_uncertainties
from gna import constructors as C

class RavelAndStripSpace(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        dest = getattr(namespace, self.dest)
        sanitized = self.__strip_spaces(values)
        if dest:
            dest.extend(sanitized)
        else:
            setattr(namespace,self.dest, sanitized)

    def __strip_spaces(self, inputs):
        ret = []
        for inp in inputs:
            ret.extend(inp.split(" "))
        return ret

class cmd(basecmd):
    pull_vararray, pull_centrals, pull_sigmas2, pull_covariance = None, None, None, None
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)
        self.opts.name = self.opts.name1 or self.opts.name

    @classmethod
    def initparser(cls, parser, env):
        name = parser.add_mutually_exclusive_group(required=True)
        name.add_argument('name', nargs='?', help='Dataset name', metavar='dataset')
        name.add_argument('-n', '--name', dest='name1', help='dataset name', metavar='dataset')

        pull = parser.add_mutually_exclusive_group()
        pull.add_argument('--pull', nargs='*', action=RavelAndStripSpace, help='Parameters to be added as pull terms')
        pull.add_argument('--pull-groups', nargs='+', help='Parameter groups to be added as pull terms')

        parser.add_argument('--theory-data', '--td', nargs=2, action='append',
                            metavar=('THEORY', 'DATA'),
                            default=[])
        parser.add_argument('--theory-data-variance', '--tdv', nargs=3, action='append',
                            metavar=('THEORY', 'DATA', 'VARIANCE'),
                            default=[])

        parser.add_argument('--error-type', choices=['pearson', 'neyman'],
                            default='pearson', help='The type of statistical errors to be used with --td')
        parser.add_argument('--variable-error', action='store_true', help='permit variable error')
        parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    def run(self):
        dataset = Dataset(desc=self.opts.name)
        verbose = self.opts.verbose
        if self.opts.verbose:
            print("Dataset '{}' with:".format(self.opts.name))

        if self.opts.pull or self.opts.pull_groups:
            self.load_pulls(dataset)

        self.snapshots = dict()
        for theory_path, data_path in self.opts.theory_data:
            try:
                theory, data = env.future['spectra', theory_path], env.future['spectra', data_path]
            except KeyError:
                theory, data = env.future['spectra', theory_path], env.future['data_spectra', data_path]
            data.data()

            if self.opts.verbose:
                print('   theory: ', str(theory))
                print('   data:   ', str(data))

            if self.opts.error_type == 'neyman':
                error=data.single()
            elif self.opts.error_type == 'pearson':
                error=theory.single()

            if not self.opts.variable_error and not error.getTaintflag().frozen():
                snapshot = self.snapshots[error] = C.Snapshot(error, labels='Snapshot: stat errors')
                snapshot.single().touch()
                error = snapshot

            dataset.assign(obs=theory, value=data, error=error.single())

        for theory_path, data_path, variance_path in self.opts.theory_data_variance:
            theory   = env.future['spectra', theory_path]
            data     = env.future['spectra', data_path]
            variance = env.future['spectra', variance_path]
            data.data()
            variance.data()

            if self.opts.verbose:
                print('   theory:  ', str(theory))
                print('   data:    ', str(data))
                print('   variance:', str(variance))

            dataset.assign(obs=theory, value=data, error=variance.single())

        self.env.parts.dataset[self.opts.name] = dataset

    def load_pulls(self, dataset):
        #
        # Load nuisance parameters
        #

        # Get list of UncertainParameter objects, drop free and fixed
        if self.opts.pull:
            pull_pars = get_parameters(self.opts.pull, drop_fixed=True, drop_free=True)
        else:
            pull_pars_dict={}
            for group in self.opts.pull_groups:
                upd = self.env.future['parameter_groups', group].unwrap()
                pull_pars_dict.update(upd)
            pull_pars = list(pull_pars_dict.values())
        variables = [par.getVariable() for par in pull_pars]
        sigmas, centrals, covariance = get_uncertainties(pull_pars)
        npars = len(pull_pars)

        print('   nuisance: {} parameters'.format(npars))

        from gna.constructors import VarArray, Points
        # Create an array, representing pull parameter values
        self.pull_vararray = VarArray(variables, labels='Nuisance: values')
        # Create an array, representing pull parameter central values
        self.pull_centrals = Points(centrals, labels='Nuisance: central')

        try:
            if covariance:
                cov = self.pull_covariance = Points(covariance, labels='Nuisance: covariance matrix')
            else:
                # If there are no correlations, store only the uncertainties
                cov = self.pull_sigmas2  = Points(sigmas**2, labels='Nuisance: sigma')
        except ValueError:
            # handle case with covariance matrix
            if covariance.any():
                cov = self.pull_covariance = Points(covariance, labels='Nuisance: covariance matrix')
            else:
                # If there are no correlations, store only the uncertainties
                cov = self.pull_sigmas2  = Points(sigmas**2, labels='Nuisance: sigma')

        dataset.assign(self.pull_vararray.single(), self.pull_centrals.single(), cov.single())

        ns = self.env.globalns('pull')
        ns.addobservable(self.opts.name, self.pull_vararray.single())
        self.env.future['pull', self.opts.name] = self.pull_vararray.single()


    __tldr__= """\
                Dataset defines:
                - A pair of theory-data:
                    * Observable (model) to be used as fitted function
                    * Observable (data) to be fitted to
                - Statistical uncertainties (Person/Neyman) [theory/observation]
                - Or nuisance parameters

                The dataset is added to the `env.future['spectra']`.

                By default a theory, fixed at the moment of dataset initialization is used for the stat errors (Pearson's case).

                Initialize a dataset 'peak' with a pair of Theory/Data:
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak_MC --nbins 50 \\
                    -- gaussianpeak --name peak_f  --nbins 50 \\
                    -- ns --name peak_MC --print \\
                          --set E0             values=2    fixed \\
                          --set Width          values=0.5  fixed \\
                          --set Mu             values=2000 fixed \\
                          --set BackgroundRate values=1000 fixed \\
                    -- ns --name peak_f --print \\
                          --set E0             values=2.5  relsigma=0.2 \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 relsigma=0.25 \\
                    -- dataset-v1 --name peak --theory-data peak_f.spectrum peak_MC.spectrum -v
                ```

                When a dataset is initialized from a nuisance terms it reads only constrained parameters from the namespace.

                Initialize a dataset 'nuisance' with a constrained parameters of 'peak_f':
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak_MC --nbins 50 \\
                    -- gaussianpeak --name peak_f  --nbins 50 \\
                    -- ns --name peak_MC --print \\
                          --set E0             values=2    fixed \\
                          --set Width          values=0.5  fixed \\
                          --set Mu             values=2000 fixed \\
                          --set BackgroundRate values=1000 fixed \\
                    -- ns --name peak_f --print \\
                          --set E0             values=2.5  relsigma=0.2 \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 relsigma=0.25 \\
                    -- dataset-v1 --name nuisance --pull peak_f -v
                ```
                """
