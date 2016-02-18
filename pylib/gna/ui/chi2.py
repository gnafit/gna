from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('inputs', type=env.parts.inputs)

    def init(self):
        chi2 = ROOT.Chi2()
        for block in self.opts.inputs:
            chi2.add(block.theory, block.data, block.cov)

        self.env.parts.statistic[self.opts.name] = chi2
        print chi2.value()
