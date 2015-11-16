from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('--reldelta', type=float, default=1.0/(65536*16))
        parser.add_argument('-a', '--add', nargs=3, action='append', default=[],
                            metavar=('NAME', 'PREDICTION', 'PAR'),
                            help='add prediction NAME for derivative of PREDICTION wrt PAR')

    def init(self):
        for name, predname, parname in self.opts.add:
            der = ROOT.Derivative(self.env.pars[parname], self.opts.reldelta)
            prediction = self.env.predictions[predname]
            der.derivative.inputs(prediction)
            out = ROOT.Prediction()
            out.append(der)
            self.env.addprediction(name, out)
            print 'Derivative', name, 'of', predname, 'wrt', parname
