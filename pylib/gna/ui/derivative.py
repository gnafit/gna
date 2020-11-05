from gna.ui import basecmd, append_typed
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--reldelta', type=float, default=1.0/(65536*16))
        parser.add_argument('-a', '--add', nargs=3, default=[],
                            metavar=('NAME', 'PREDICTION', 'PAR'),
                            action=append_typed(str, env.parts.prediction, str),
                            help='add prediction NAME for derivative of PREDICTION wrt PAR')

    def init(self):
        for name, prediction, parname in self.opts.add:
            der = ROOT.Derivative(self.env.pars[parname], self.opts.reldelta)
            der.derivative.inputs(prediction)
            out = ROOT.Concat()
            out.add(der)
            self.env.parts.prediction[name] = out
