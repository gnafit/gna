from gna.ui import basecmd
import ROOT
import argparse
import numpy as np

class CovmatAppendAction(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 3:
            msg = 'expected at least three arguments'
            raise argparse.ArgumentError(self, msg)
        newvalues = (values[0], values[1], values[2:])
        super(CovmatAppendAction, self).__call__(parser, namespace, newvalues, option_string)

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('-a', '--add', nargs=2, action='append', default=[],
                            metavar=('NAME', 'DIAGONAL'),
                            help='add statistical covariance matrix NAME from DIAGONAL')
        parser.add_argument('-s', '--systematics', nargs='+', action=CovmatAppendAction, default=[],
                            metavar=('NAME PREDICTION PAR', 'PAR'),
                            help='add to covariance matrix NAME systematics of PREDICTION wrt PARs')
        parser.add_argument('-f', '--fix', action='append', default=[],
                            help='fix covariance matrix NAME')
        parser.add_argument('-p', '--print', action='append', default=[],
                            metavar='NAME',
                            help='print covariance matrix NAME')

    def init(self):
        for name, diagobj in self.opts.add:
            covmat = ROOT.Covmat()
            covmat.cov.stat.connect(self.env.get(diagobj))
            covmat.inv.cov(covmat.cov)
            self.env.addcovmat(name, covmat)
            print 'Covmat', name, 'from', diagobj

        for name, predname, pars in self.opts.systematics:
            prediction = self.env.predictions[predname]
            covmat = self.env.covmats[name]
            for parname in pars:
                der = ROOT.Derivative(self.env.pars[parname])
                der.derivative.inputs(prediction)
                covmat.rank1(der)

        for name in self.opts.fix:
            self.env.covmats[name].setFixed(True)

        for name in getattr(self.opts, 'print'):
            covmat = self.env.covmats[name]
            print name
            print np.frombuffer(covmat.data(), count=covmat.size()).reshape((covmat.ndim(), covmat.ndim()))
