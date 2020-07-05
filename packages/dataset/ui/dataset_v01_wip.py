# -*- coding: utf-8 -*-
"""Dataset initialization (v01 WIP). Configures the dataset for a single experiment.
Dataset defines:
    - Observable (model) to be used as fitted function
    - Observable (data) to be fitted to
    - Statistical uncertainty (Person/Neyman) [theory/observation]
    - Nuisance parameters
    """
from __future__ import print_function
from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from gna.env import env
from itertools import chain
from gna.dataset import Dataset
from gna.parameters.parameter_loader import get_parameters, get_uncertainties
from gna import constructors as C

class cmd(basecmd):
    pull_vararray, pull_centrals, pull_sigmas2, pull_covariance = None, None, None, None

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True, help='Dataset name', metavar='dataset')

        pull = parser.add_mutually_exclusive_group()
        pull.add_argument('--pull', action='append', help='Parameters to be added as pull terms')

        parser.add_argument('--asimov-data', nargs=2, action='append',
                            metavar=('THEORY', 'DATA'),
                            default=[])
        # parser.add_argument('--asimov-poisson', nargs=2, action='append',
                            # metavar=('THEORY', 'DATA'),
                            # default=[])
        parser.add_argument('--error-type', choices=['pearson', 'neyman'],
                            default='pearson', help='The type of statistical errors to be used')
        parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    def run(self):
        dataset = Dataset(desc=None)
        verbose = self.opts.verbose
        if verbose:
            print('Adding pull parameters to dataset', self.opts.name)

        if self.opts.pull:
            self.load_pulls(dataset)

        self.snapshots = dict()
        if self.opts.asimov_data:
            for theory_path, data_path in self.opts.asimov_data:
                try:
                    theory, data = env.get(theory_path), env.get(data_path)
                except KeyError:
                    theory, data = env.future['spectra', theory_path], env.future['spectra', data_path]

                if self.opts.error_type == 'neyman':
                    error=data.single()
                elif self.opts.error_type == 'pearson':
                    error=theory.single()

                if not error.getTaintflag().frozen():
                    snapshot = self.snapshots[error] = C.Snapshot(error, labels='Snapshot: stat errors')
                    snapshot.single().touch()
                    error = snapshot

                dataset.assign(obs=theory, value=data, error=error.single())

        # if self.opts.asimov_poisson:
            # for theory_path, data_path in self.opts.asimov_poisson:
                # data_poisson = np.random.poisson(env.get(data_path).data())
                # if self.opts.error_type == 'neyman':
                    # dataset.assign(env.get(theory_path),
                                   # data_poisson,
                                   # env.get(data_path))
                # elif self.opts.error_type == 'pearson':
                    # dataset.assign(env.get(theory_path),
                                   # data_poisson,
                                   # env.get(theory_path))

        self.env.parts.dataset[self.opts.name] = dataset

    def load_pulls(self, dataset):
        #
        # Load nuisance parameters
        #

        # Get list of UncertainParameter objects, drop free and fixed
        pull_pars = get_parameters(self.opts.pull, drop_fixed=True, drop_free=True)
        variables = [par.getVariable() for par in pull_pars]
        sigmas, centrals, covariance = get_uncertainties(pull_pars)
        npars = len(pull_pars)

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
