# -*- coding: utf-8 -*-
from __future__ import absolute_import
from gna.ui import basecmd, append_typed, at_least
import ROOT

class cmd(basecmd):
    @classmethod
    def initparser(self, parser, env):
        parser.add_argument('name')
        parser.add_argument('-p', '--parameters', nargs='+', default=[],
                            metavar=('PAR'),
                            help='include covariance due to parameters PARs')

    def init(self):
        covariance = {
            'covariated': [],
            'parameters': [],
        }
        self.env.parts.covariance[self.opts.name] = covariance

        covariance['covariated'].extend(self.env.covariances)

        for pname in self.opts.parameters:
            covariance['parameters'].append(self.env.pars[pname])
