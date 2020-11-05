#
"""Build test statistic based on χ² function"""

from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='statistic name')
        parser.add_argument('analysis', type=env.parts.analysis)

    def init(self):
        chi2 = ROOT.Chi2()
        for block in self.opts.analysis:
            chi2.add(block.theory, block.data, block.cov)

        self.env.parts.statistic[self.opts.name] = chi2
