from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('analysis', type=env.parts.analysis)
        parser.add_argument('--ln-approx', action='store_true', default=False)

    def init(self):
        poisson = ROOT.Poisson(self.opts.ln_approx)
        for block in self.opts.analysis:
            poisson.add(block.theory, block.data)


        self.env.parts.statistic[self.opts.name] = poisson


