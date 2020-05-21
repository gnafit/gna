# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from gna.ui import basecmd, append_typed, at_least, qualified
import ROOT
import argparse
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('-a', '--add', nargs=2, default=[],
                            metavar=('NAME', 'DIAGONAL'),
                            action=append_typed(str, qualified(env.parts.prediction, env.parts.data)),
                            help='add statistical covariance matrix NAME from DIAGONAL')
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
            print('Covmat', name, 'from', diag)


        for covmat in self.opts.fix:
            covmat.setFixed(True)

        for covmat in getattr(self.opts, 'print'):
            print(covmat.data())
