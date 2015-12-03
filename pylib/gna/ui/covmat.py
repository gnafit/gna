from gna.ui import basecmd, append_typed, at_least, qualified
import ROOT
import argparse
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-a', '--add', nargs=2, default=[],
                            metavar=('NAME', 'DIAGONAL'),
                            action=append_typed(str, qualified(env.parts.prediction, env.parts.data)),
                            help='add statistical covariance matrix NAME from DIAGONAL')
        parser.add_argument('-s', '--systematics', nargs='+', default=[],
                            metavar=('NAME PREDICTION PAR', 'PAR'),
                            action=append_typed(str, env.parts.prediction, at_least(1, str)),
                            help='add to covariance matrix NAME systematics of PREDICTION wrt PARs')
        parser.add_argument('-f', '--fix', default=[],
                            action=append_typed(env.parts.covmat, lazy=True),
                            help='fix covariance matrix NAME')
        parser.add_argument('-p', '--print', default=[],
                            metavar='NAME',
                            action=append_typed(env.parts.covmat, lazy=True),
                            help='print covariance matrix NAME')

    def init(self):
        for name, diag in self.opts.add:
            covmat = ROOT.Covmat()
            covmat.cov.stat.connect(diag)
            covmat.cholesky.cov(covmat.cov)
            self.env.parts.covmat[name] = covmat
            print 'Covmat', name, 'from', diag

        for name, prediction, pars in self.opts.systematics:
            covmat = self.env.parts.covmat[name]
            for parname in pars:
                der = ROOT.Derivative(self.env.pars[parname])
                der.derivative.inputs(prediction)
                covmat.rank1(der)

        for covmat in self.opts.fix:
            covmat.setFixed(True)

        for covmat in getattr(self.opts, 'print'):
            print np.frombuffer(covmat.data(), count=covmat.size()).reshape((covmat.ndim(), covmat.ndim()))
