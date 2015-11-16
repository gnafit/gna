from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('prediction')
        parser.add_argument('data')
        parser.add_argument('covmat')

    def init(self):
        prediction = self.env.predictions[self.opts.prediction]
        data = self.env.data[self.opts.data]
        covmat = self.env.covmats[self.opts.covmat]

        chi2 = ROOT.Chi2()
        chi2.chi2.prediction(prediction)
        chi2.chi2.data(data)
        chi2.chi2.invcov(covmat.inv)

        print 'chi2:', chi2.value()
