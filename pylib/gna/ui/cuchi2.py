from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('analysis', type=env.parts.analysis)

    def init(self):
        cuchi2 = ROOT.CuChi2()
        for block in self.opts.analysis:
            cuchi2.add(block.theory, block.data, block.cov)

        self.env.parts.statistic[self.opts.name] = cuchi2
