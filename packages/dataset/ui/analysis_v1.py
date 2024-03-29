"""Analysis module (v1) combines multiple datasets for the analysis (fit). May provide a covariance matrix based on par group."""

from gna.ui import basecmd, append_typed, at_least
import ROOT
from gna.dataset import Dataset
from gna import constructors as C
from gna.parameters.parameter_loader import get_parameters

class analysis_v1(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)
        self.dataset = None
        self.opts.name = self.opts.name1 or self.opts.name

        if self.opts.observables:
            if len(self.opts.observables)!=len(self.opts.datasets):
                raise Exception("Number of observables should be the same as number of datasets or 0")

    @classmethod
    def initparser(cls, parser, env):
        name = parser.add_mutually_exclusive_group(required=True)
        name.add_argument('name', nargs='?', help='Dataset name', metavar='dataset')
        name.add_argument('-n', '--name', dest='name1', help='analysis name', metavar='analysis')

        parser.add_argument('-d', '--datasets', nargs='+', required=True,
                            type=env.parts.dataset,
                            metavar='dataset', help='datasets to use')
        parser.add_argument('-p', '--cov-parameters', metavar='pargroup', help='parameters for the covariance matrix')
        parser.add_argument('--cov-strict', action='store_true', help='raise exception in case there are skept parameters')
        parser.add_argument('-o', '--observables', nargs='+',
                            metavar='observable', help='observables (model) to be fitted')
        parser.add_argument('--toymc', choices=['covariance', 'poisson', 'normal', 'normalStats', 'asimov'], help='use random sampling to variate the data/theory', metavar='fluct')
        parser.add_argument('--toymc-source', choices=('data', 'theory'), default='theory', help='the source for the --toymc option', metavar='source')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbose mode')
        parser.add_argument('--covariance-updater', metavar="HOOK_NAME",
                            help='Name of hook that triggers covariance matrix update')

    def __extract_obs(self, obses):
        for obs in obses:
            if '/' in obs:
                yield self.env.get(obs)
            elif '.' in obs:
                yield self.env.future[(obs)]
            else:
                for param in get_parameters([obs], drop_fixed=True, drop_free=True):
                    yield param

    def run(self):
        dataset = self.dataset = Dataset(bases=list(reversed(self.opts.datasets)), desc=self.opts.name)

        if self.opts.cov_parameters:
            try:
                cov_parameters = self.env.future['parameter_groups', self.opts.cov_parameters]
            except KeyError:
                raise Exception('Unable to get pargroup {}'.format(self.opts.cov_parameters))

            cov_parameters_all = list(cov_parameters.values())
            cov_parameters, skip_parameters = partition(lambda par: par_influences(par, dataset.data.keys()), cov_parameters_all)
            if skip_parameters:
                print('Skip {} cov parameters as they do not affect the model'.format(len(skip_parameters)))
                if self.opts.verbose>1:
                    print(' ', [p.qualifiedName() for p in skip_parameters])
                if self.opts.cov_strict:
                    raise self._exception('Some parameters do not affect the model.')
        else:
            cov_parameters = []
            skip_parameters = []

        if self.opts.observables:
            observables = list(self.__extract_obs(self.opts.observables))
        else:
            observables = None

        if self.opts.verbose:
            names = ', '.join((d.desc for d in self.opts.datasets))
            print("Analysis '{}' with: {}".format(self.opts.name, names), end='')
            if self.opts.cov_parameters:
                print(f' and {len(cov_parameters)} parameters from {self.opts.cov_parameters}', end='')
                if skip_parameters:
                    print(f' (skip {len(skip_parameters)})')
            else:
                print()
            if self.opts.verbose>1 and cov_parameters:
                print(' ', [p.qualifiedName() for p in cov_parameters])

        blocks = dataset.makeblocks(observables, cov_parameters)

        if self.opts.toymc:
            if self.opts.toymc == 'covariance':
                toymc = ROOT.CovarianceToyMC()
                add = toymc.add
            elif self.opts.toymc == 'poisson':
                toymc = ROOT.PoissonToyMC()
                add = lambda t, c: toymc.add(t)
            elif self.opts.toymc == 'normal':
                toymc = C.NormalToyMC()
                add = toymc.add
            elif self.opts.toymc == 'normalStats':
                toymc = C.NormalStatsToyMC()
                add = toymc.add
            elif self.opts.toymc == 'asimov':
                toymc = C.Snapshot()
                add = lambda t, _: toymc.add_input(t)
            else:
                assert False

            if self.opts.toymc_source=='data':
                for block in blocks:
                    add(block.data, block.cov)
            elif self.opts.toymc_source=='theory':
                for block in blocks:
                    add(block.theory, block.cov)
            else:
                assert False

            blocks = [ block._replace(data=toymc_out)
                      for (block, toymc_out) in zip(blocks, toymc.transformations.front().outputs.values()) ]

            self.env.parts.toymc[self.opts.name] = toymc
            for toymc in toymc.transformations.values():
                toymc.setLabel(self.opts.toymc+' ToyMC '+self.opts.name)

        self.env.parts.analysis[self.opts.name] = blocks
        self.env.parts.analysis_errors[self.opts.name] = dataset

        storage = self.env.future.child(('analysis', self.opts.name))
        for i, block in enumerate(blocks):
            istr = str(i)

            covbase = dataset.predictions[i].covbase.covbase
            storage[istr] = dict(theory=block.theory, data=block.data, L=block.cov.single(), covbase=covbase)

        update_covariance_set = self.env.future.setdefault(('hooks', self.opts.name, 'covariance'), self.update_covariance)
        assert self.update_covariance==update_covariance_set, 'May not overwrite covariance hook'

    def update_covariance(self):
        self.dataset.reset_errors()

def par_influences(parameter, observables):
    for observable in observables:
        if parameter.influences(observable):
            return True
    return False

def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses

analysis_v1.__tldr__ =  """\
                Creates a named analysis, i.e. a triplet of theory, data and covariance matrix. The covariance matrix
                may be diagonal and contain only statistical uncertainties or contain a systematic part as well.

                The `analysis-v1` required a name and a few of datasets after `-d` option.

                Initialize an analysis 'analysis' with a dataset 'peak':
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
                    -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum -v \\
                    -- analysis-v1 --name analysis --datasets peak -v
                ```

                Initialize an analysis 'analysis' with a dataset 'peak' and covariance matrix based on constrained parameters:
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
                    -- dataset-v1 peak --theory-data peak_f.spectrum peak_MC.spectrum -v \\
                    -- pargroup covpars peak_f -m constrained \\
                    -- analysis-v1  analysis --datasets peak -p covpars -v
                ```
                """
