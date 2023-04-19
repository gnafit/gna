"""Initializes a minimizer for a given statistic and set of parameters.

Based on `minimizer-v2`:
    - default minimizer is iminuit
    - default starting point is `value`
    - works with stats-v1 at least
"""

import ROOT
from gna.ui import basecmd, set_typed
from minimize.lib import minimizers
from minimize.lib.minpars import MinPars
import yaml

def yaml_load(s):
    return yaml.load(s, Loader=yaml.Loader) or {}

class minimizer_v2(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='Minimizer name')
        parser.add_argument('statistic', help='Statistic name', metavar='statmodule')
        parser.add_argument('pargroups', nargs='+', default=[], help='Parameters groups to minimize with')

        minimizer = parser.add_argument_group(title='minimizer')
        minimizer.add_argument('-t', '--type', choices=minimizers.keys(), default='iminuit',
                                    help='Minimizer type {%(choices)s}', metavar='minimizer')
        minimizer.add_argument('--minopts', type=yaml_load, help='Options to pass to the minimizer')

        parser.add_argument('-s', '--strict', action='store_true', help='raise exception if some parameters are skipped')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')

        parser.add_argument('--initial-value', default='value', choices=('central', 'value'), help='define what initial value to use')

    def init(self):
        try:
            output = self.env.future['statistic'][self.opts.statistic]['fun']
        except KeyError:
            raise self._exception(f'Unable to find statistic {self.opts.statistic} in env.future.statistic')
        self.statistic = ROOT.StatisticOutput(output)
        self.minpars={}
        for pargroup in self.opts.pargroups:
            self.minpars.update(self.env.future['parameter_groups', pargroup].unwrap())

        initial_central = self.opts.initial_value=='central'
        self.minpars = MinPars(self.minpars, check=self.statistic.output(), initial_central=initial_central)
        if self.minpars._skippars:
            print('Minimizer {}: skip {} parameters not affecting the function'.format(self.opts.name, len(self.minpars._skippars)))

            if self.opts.strict:
                raise self._exception('Some parameters are skipped')

        if self.opts.verbose>1:
            if self.minpars._skippars:
                print('Skip {} parameters:'.format(len(self.minpars._skippars)), [p.qualifiedName() for p in self.minpars._skippars])
            print('Minimizer {} parameters:'.format(self.opts.name))
            self.minpars.dump()

        minopts = {'minimizable_verbose': self.opts.verbose>2}
        if self.opts.minopts:
            minopts.update(self.opts.minopts)
        self.minimizer = minimizers[self.opts.type](self.statistic, self.minpars, name=self.opts.name, **minopts)

        self.env.future[('minimizer', self.opts.name)] = self.minimizer

    __tldr__ =  """\
                The module creates a minimizer instance which then may be used for a fit with `fit-v1` module or elsewhere.
                The minimizer arguments are: `minimizer name` `statistics` `minpars`. Where:
                * `minimizer name` is a name of new minimizer.
                * `statistics` is the name of a function to minimizer, which should be created beforehand.
                * `minpars` is the name of a parameter group, created by `pargroup`.

                The minimizer is stored in `env.future['minimizer']` under its name.

                Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars':
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
                    -- stats-v1 stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- minimizer-v2 min stats minpars -vv \\
                    -- fit-v1 min \\
                    -- env-print fitresult.min
                ```
                The `env-print` will print the status of the minimization, performed by the `fit-v1`.

                By default `TMinuit2` minimizer is used from ROOT. The minimizer may be changed with `-t` option to
                `scipy` or `minuit` (TMinuit).

                Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars' using a `scipy` minimizer:
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
                    -- stats-v1 stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- minimizer-v2 min stats minpars -vv -t scipy \\
                    -- fit-v1 min \\
                    -- env-print fitresult.min
                ```

                The module is based on `minimizer` and completely supersedes it.

                See also: `minimizer-scan`, `fit-v1`, `stats-v1`, `pargroup`.
                """
