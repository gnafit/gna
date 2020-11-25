"""Initializes a minimizer for a given statistic and set of parameters."""

import ROOT
from gna.ui import basecmd, set_typed
from packages.minimize.lib import minimizers
from packages.minimize.lib.minpars import MinPars

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='Minimizer name')
        parser.add_argument('statistic', action=set_typed(env.parts.statistic), help='Statistic name',
                                         metavar='statmodule')
        parser.add_argument('pargroup', help='Parameters group to minimize')

        parser.add_argument('-t', '--type', choices=minimizers.keys(), default='minuit2',
                                    help='Minimizer type {%(choices)s}', metavar='minimizer')

        parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')

    def init(self):
        self.statistic = ROOT.StatisticOutput(self.opts.statistic.transformations.back().outputs.back())
        self.minpars = self.env.future['parameter_groups'][self.opts.pargroup]
        self.minpars = MinPars(self.minpars)
        if self.opts.verbose>1:
            print('Minimizer {} parameters:'.format(self.opts.name))
            self.minpars.dump()
        self.minimizer = minimizers[self.opts.type](self.statistic, self.minpars, name=self.opts.name)

        self.env.future[('minimizer', self.opts.name)] = self.minimizer

    __tldr__ =  """\
                The module creates a minimizer instance which then may be used for a fit with `fit-v1` module or elsewhere.
                The minimizer arguments are: `minimizer name` `statistics` `minpars`. Where:
                * `minimizer name` is a name of new minimizer.
                * `statistics` is the name of a function to minimizer, which should be created beforehand.
                * `minpars` is the name of a parameter group, created by `pargroup`.

                \033[32mCreate a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars':
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
                    -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum \\
                    -- analysis-v1 --name analysis --datasets peak \\
                    -- stats stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- minimizer-v1 min stats minpars -vv \\
                    -- fit-v1 min \\
                    -- env-print fitresult.min
                ```
                The `env-print` will print the status of the minimization, performed by the `fit-v1`.

                By default `TMinuit2` minimizer is used from ROOT. The minimizer may be changed with `-t` option to
                `scipy` or `minuit` (TMinuit).

                \033[32mCreate a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars' using a `scipy` minimizer:
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
                    -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum \\
                    -- analysis-v1 --name analysis --datasets peak \\
                    -- stats stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- minimizer-v1 min stats minpars -vv \\
                    -- fit-v1 min \\
                    -- env-print fitresult.min
                ```

                The module is based on `minimizer` and completely supersedes it.

                See also: `minimizer-scan`, `fit-v1`, `stats`, `pargroup`.
                """
