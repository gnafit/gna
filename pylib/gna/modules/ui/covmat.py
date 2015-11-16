from gna.ui import basecmd
import ROOT
import argparse
import numpy as np
from matplotlib import pyplot as plt

class CovmatAppendAction(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2:
            msg = 'expected at least two arguments'
            raise argparse.ArgumentError(self, msg)
        newvalues = (values[0], values[1], values[2:])
        super(CovmatAppendAction, self).__call__(parser, namespace, newvalues, option_string)

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('-a', '--add', nargs='+', action=CovmatAppendAction, default=[],
                            metavar=('NAME PREDICTION', 'PAR'),
                            help='add prediction NAME for observables specified by OBSSPEC (o1[+o2+[...]])')
        parser.add_argument('-p', '--print', action='append', default=[],
                            metavar='NAME',
                            help='print prediction NAME')


    def init(self):
        for name, predname, pars in self.opts.add:
            prediction = self.env.predictions[predname]
            covmat = ROOT.Covmat()
            covmat.cov.stat.connect(prediction)
            for parname in pars:
                der = ROOT.Derivative(self.env.pars[parname])
                der.derivative.inputs(prediction)
                covmat.rank1(der)
            self.env.addcovmat(name, covmat)
            print 'Covmat', name, 'for', predname, 'wrt', ', '.join(pars)

        for name in getattr(self.opts, 'print'):
            covmat = self.env.covmats[name]
            print name
            print np.frombuffer(covmat.data(), count=covmat.size()).reshape((covmat.ndim(), covmat.ndim()))
