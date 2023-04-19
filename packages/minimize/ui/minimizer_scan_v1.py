"""Initializes a hybrid minimizer which does a raster scan over a part of the variables.

Based on `minimizer-scan`:
    - add options from `minimizer-v2`
        - `--minopts` to pass the options to the minimizer
        - `--initial-value` to choose the minimization start
        - work with stats-v1
    - use `--pargrid` to specify the parameter grid
"""

import ROOT
from gna.ui import basecmd, set_typed
from minimize.lib import minimizers
from minimize.lib.minpars import MinPars
import yaml
from typing import Mapping, List, Any
from collections import abc
from minimize.lib.scanminimizer import ScanMinimizer

def yaml_load(s):
    return yaml.load(s, Loader=yaml.Loader) or {}

class minimizer_scan_v1(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='Minimizer name')
        parser.add_argument('statistic', help='Statistic name', metavar='statmodule')
        parser.add_argument('pargroups', nargs='+', default=[], help='Parameters groups to minimize with')
        parser.add_argument('--pargrid',  required=True, help='Parameters grid to scan')

        minimizer = parser.add_argument_group(title='minimizer')
        minimizer.add_argument('-t', '--type', choices=minimizers.keys(), default='iminuit',
                                    help='Minimizer type {%(choices)s}', metavar='minimizer')
        minimizer.add_argument('--minopts', type=yaml_load, help='Options to pass to the minimizer')

        parser.add_argument('-s', '--strict', action='store_true', help='raise exception if some parameters are skipped')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')

        parser.add_argument('--initial-value', default='value', choices=('central', 'value'), help='define what initial value to use')

    def init(self):
        def extract_names(obj: Mapping[str, Any], level=0) -> List[str]:
            names = []
            for key, val in obj.items():
                if isinstance(val, abc.Mapping):
                    recurse = [".".join((key, name)) for name in extract_names(val)]
                    names.extend(recurse)
                else:
                    names.append(key)
            return names
        def strip_internal(name: str)->str:
            for stub in ['.par', '.grid']:
                if name.endswith(stub):
                    return name[:-len(stub)]


        try:
            output = self.env.future['statistic'][self.opts.statistic]['fun']
        except KeyError:
            raise self._exception(f'Unable to find statistic {self.opts.statistic} in env.future.statistic')
        self.statistic = ROOT.StatisticOutput(output)
        self.minpars={}
        for pargroup in self.opts.pargroups:
            self.minpars.update(self.env.future['parameter_groups', pargroup].unwrap())
        self.gridpars = self.env.future['pargrid'][self.opts.pargrid]
        gridpar_names = [strip_internal(name) for name in extract_names(self.gridpars.unwrap())]
        gridpar_names = list(dict.fromkeys(gridpar_names))

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
        minimizerclass = minimizers[self.opts.type]
        self.minimizer = ScanMinimizer(self.statistic, self.minpars, self.gridpars, minimizerclass,
                                       name=self.opts.name, fixed_order=gridpar_names,
                                       verbose=bool(self.opts.verbose))
        self.env.future[('minimizer', self.opts.name)] = self.minimizer

    __tldr__ =  """\
                The hybrid minimizer minimizes a set of parameters simply by scanning them, all the other parameters
                are minimized via regular minimizer at each point.
                After the best fit is found, the minimizer performs a minimization over all the parameters.
                The structure is similar with the `minimizer-v1`.

                The module creates a minimizer instance which then may be used for a fit with `fit-v1` module or elsewhere.
                The minimizer arguments are: `minimizer name` `statistics` `minpars` and `gridpars`. Where:
                * `minimizer name` is a name of new minimizer.
                * `statistics` is the name of a function to minimizer, which should be created beforehand.
                * `minpars` is the name of a parameter group, created by `pargroup`.
                * `gridpars` is the name of a parameter group, created by `pargrid`.
                  It is important to note, that the grid parameters should also be included in the `minpars` group.

                The minimizer is stored in `env.future['minimizer']` under its name.

                Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars', but do a raster scan over E0 (linear) and Width (log):
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
                    -- dataset-v1 peak --theory-data peak_f.spectrum peak_MC.spectrum \\
                    -- analysis-v1 analysis --datasets peak \\
                    -- stats-v1 stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- pargrid  scangrid --linspace  peak_f.E0    0.5 4.5 10 \\
                                         --geomspace peak_f.Width 0.3 0.6 5 -v \\
                    -- minimizer-scan min stats minpars --pargrid scangrid -vv \\
                    -- fit-v1 min -p --push \\
                    -- env-print fitresult.min
                ```
                The `env-print` will print the status of the minimization, performed by the `fit-v1`.
                The intermediate results are saved in 'fitresults'.

                By default `TMinuit2` minimizer from ROOT is used. The minimizer may be changed with `-t` option to
                `scipy` or `minuit` (TMinuit).

                The module is based on `minimizer` and completely supersedes it.

                See also: `minimizer-v1`, `fit-v1`, `stats`, `pargroup`.
                """
