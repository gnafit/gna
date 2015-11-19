from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('prediction', type=env.parts.prediction)
        parser.add_argument('data', type=env.parts.data)
        parser.add_argument('covmat', type=env.parts.covmat)

    def init(self):
        chi2 = ROOT.Chi2()
        chi2.chi2.prediction(self.opts.prediction)
        chi2.chi2.data(self.opts.data)
        chi2.chi2.invcov(self.opts.covmat.inv)

        self.env.parts.statistic[self.opts.name] = chi2
