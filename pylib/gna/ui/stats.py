# encoding: utf-8

u"""Build test statistic based on arbitrary sum of χ² and lnPoisson functions"""

from gna.ui import basecmd
import ROOT
import numpy as np
from gna import constructors as C

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='statistic name')
        parser.add_argument('-c', '--chi2',       default=[], action='append', type=env.parts.analysis, help=u'χ² contribution')
        parser.add_argument('-p', '--lnpoisson',  default=[], action='append', type=env.parts.analysis, help=u'lnPoisson contribution')
        parser.add_argument('--lnpoisson-legacy', default=[], action='append', type=env.parts.analysis, help=u'lnPoisson contribution (legacy)')
        parser.add_argument('--poisson-approx', action='store_true', help='Use approximate lnPoisson formula (Stirling)')

    def init(self):
        self.components = []

        if self.opts.chi2:
            self.chi2 = ROOT.Chi2()
            for analysis in self.opts.chi2:
                for block in analysis:
                    self.chi2.add(block.theory, block.data, block.cov)

            self.components.append(self.chi2)

        if self.opts.lnpoisson_legacy:
            self.lnpoisson_legacy = ROOT.Poisson(self.opts.poisson_approx)
            for analysis in self.opts.lnpoisson_legacy:
                for block in analysis:
                    self.lnpoisson_legacy.add(block.theory, block.data)

            self.components.append(self.lnpoisson_legacy)

        if self.opts.lnpoisson:
            self.lnpoisson = ROOT.LnPoissonSplit(self.opts.poisson_approx)
            for analysis in self.opts.lnpoisson:
                for block in analysis:
                    self.lnpoisson.add(block.theory, block.data)

            self.components.append(self.lnpoisson)

        if len(self.components)==1:
            self.statistic = self.components[0]
        else:
            self.statistic = C.Sum([comp.single() for comp in self.components])

        self.env.parts.statistic[self.opts.name] = self.statistic
