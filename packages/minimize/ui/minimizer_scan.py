"""Minimizer-scan module: hybrid scan/quasi newton minimizer

Minimizes a set of parameters simply scanning them, all others are minimized vie regular minimizer at each point
"""

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
        parser.add_argument('pargrid',  help='Parameters grid to scan')

        parser.add_argument('-t', '--type', choices=minimizers.keys(), default='minuit2',
                                    help='Minimizer type {%(choices)s}', metavar='minimizer')

        parser.add_argument('-v', '--verbose', action='count', help='increase verbosity level')

    def init(self):
        self.statistic = ROOT.StatisticOutput(self.opts.statistic.transformations.back().outputs.back())
        self.minpars = self.env.future['parameter_groups'][self.opts.pargroup]
        self.minpars = MinPars(self.minpars)
        if self.opts.verbose>1:
            print('Minimizer {} parameters:'.format(self.opts.name))
            self.minpars.dump()
        self.minimizer = minimizers[self.opts.type](self.statistic, self.minpars)

        self.env.future[('minimizer', self.opts.name)] = self.minimizer
