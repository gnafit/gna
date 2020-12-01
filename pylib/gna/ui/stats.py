"""Build test statistic based on arbitrary sum of χ² and logPoisson functions"""


from gna.ui import basecmd
import ROOT
import numpy as np
from gna import constructors as C

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='statistic name')
        parser.add_argument('-c', '--chi2',       default=[], action='append', type=env.parts.analysis, help='χ² contribution')
        parser.add_argument('-p', '--logpoisson',  default=[], action='append', type=env.parts.analysis, help='logPoisson contribution')
        parser.add_argument('--logpoisson-legacy', default=[], action='append', type=env.parts.analysis, help='logPoisson contribution (legacy)')
        parser.add_argument('--poisson-approx', action='store_true', help='Use approximate logPoisson formula (Stirling)')
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

        if self.opts.logpoisson_legacy:
            clabel = labels and labels.pop() or ''
            self.logpoisson_legacy = ROOT.Poisson(self.opts.poisson_approx, labels='')
            for analysis in self.opts.logpoisson_legacy:
                for block in analysis:
                    self.logpoisson_legacy.add(block.theory, block.data)

            self.components.append(self.logpoisson_legacy)

        if self.opts.logpoisson:
            clabel = labels and labels.pop() or ''
            self.logpoisson = ROOT.LogPoissonSplit(self.opts.poisson_approx, labels=(clabel+' (const)', clabel))
            for analysis in self.opts.logpoisson:
                for block in analysis:
                    self.logpoisson.add(block.theory, block.data)

            self.components.append(self.logpoisson)

        if len(self.components)==1:
            self.statistic = self.components[0]
        else:
            clabel = labels and labels.pop() or ''
            self.statistic = C.Sum([comp.transformations.back() for comp in self.components], labels=clabel)

        self.env.parts.statistic[self.opts.name] = self.statistic
