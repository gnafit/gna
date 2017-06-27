from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('analysis', type=env.parts.analysis)
        parser.add_argument('--ln_approx', action='store_true', default=False)

    def init(self):
        poisson = ROOT.Poisson()
        for block in self.opts.analysis:
            poisson.add(block.theory, block.data, block.cov, self.opts.ln_approx)	# can be without 4th arg, default value is false


        self.env.parts.statistic[self.opts.name] = poisson


