"""Dataset initialization. Configures the dataset for a single experiment.
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
from gna.parameters.parameter_loader import get_parameters
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
        parser.add_argument('--random-seed', type=int, help='Set random seed of numpy random generator to given value')
        parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    def run(self):
        if self.opts.random_seed:
            np.random.seed(self.opts.random_seed)

        dataset = Dataset(desc=None)
        verbose = self.opts.verbose
        if verbose:
            print('Adding pull parameters to dataset', self.opts.name)

        if self.opts.pull:
            self.load_pulls(dataset)

        self.snapshots = dict()
        if self.opts.asimov_data:
            for theory_path, data_path in self.opts.asimov_data:
                theory, data = env.get(theory_path), env.get(data_path)
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
        npars = len(pull_pars)

        variables, centrals, sigmas = [None]*npars, np.zeros(npars, dtype='d'), np.zeros(npars, dtype='d')
        correlations = False

        # Get variables, central values and sigmas, check that there are (no) correlations
        for i, par in enumerate(pull_pars):
            variables[i]=par.getVariable()
            centrals[i]=par.central()
            sigmas[i]=par.sigma()
            correlations = correlations or par.isCorrelated()

        from gna.constructors import VarArray, Points
        # Create an array, representing pull parameter values
        self.pull_vararray = VarArray(variables, labels='Nuisance: values')
        # Create an array, representing pull parameter central values
        self.pull_centrals = Points(centrals, labels='Nuisance: central')

        if correlations:
            # In case there are correlations:
            # - create covariance matrix
            # - fill the diagonal it with the value of sigma**2
            # - fill the off-diagonal elements with covarainces
            # - create Points, representing the covariance matrix
            covariance = np.diag(sigma**2)
            for i in range(npars):
                for j in range(i):
                    pari, parj = pull_pars[i], pull_pars[j]
                    cov = pari.getCovariance(parj)
                    covariance[i,j]=covariance[j,i]=cov

            self.pull_covariance = Points(covariance, labels='Nuisance: covariance matrix')
        else:
            # If there are no correlations, store only the uncertainties
            self.pull_sigmas2  = Points(sigmas**2, labels='Nuisance: sigma')

        dataset.assign(self.pull_vararray.single(), self.pull_centrals.single(), self.pull_sigmas2.single())

        ns = self.env.globalns('pull')
        ns.addobservable(self.opts.name, self.pull_vararray.single())


