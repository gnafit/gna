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
        parser.add_argument('--labels', nargs='+', default=[], help='Node labels')

    def init(self):
        self.components = []
        labels = list(reversed(self.opts.labels))

        if self.opts.chi2:
            clabel = labels and labels.pop() or ''
            self.chi2 = ROOT.Chi2(labels=clabel)
            for analysis in self.opts.chi2:
                for block in analysis:
                    self.chi2.add(block.theory, block.data, block.cov)

            self.components.append(self.chi2)

        if self.opts.lnpoisson_legacy:
            clabel = labels and labels.pop() or ''
            self.lnpoisson_legacy = ROOT.Poisson(self.opts.poisson_approx, labels='')
            for analysis in self.opts.lnpoisson_legacy:
                for block in analysis:
                    self.lnpoisson_legacy.add(block.theory, block.data)

            self.components.append(self.lnpoisson_legacy)

        if self.opts.lnpoisson:
            clabel = labels and labels.pop() or ''
            self.lnpoisson = ROOT.LnPoissonSplit(self.opts.poisson_approx, labels=(clabel+' (const)', clabel))
            for analysis in self.opts.lnpoisson:
                for block in analysis:
                    self.lnpoisson.add(block.theory, block.data)

            self.components.append(self.lnpoisson)

        if len(self.components)==1:
            self.statistic = self.components[0]
        else:
            clabel = labels and labels.pop() or ''
            self.statistic = C.Sum([comp.transformations.back() for comp in self.components], labels=clabel)

        self.env.parts.statistic[self.opts.name] = self.statistic
