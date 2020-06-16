# -*- coding: utf-8 -*-
"""A group of parameters UI: select a set of parameters and store"""

from __future__ import print_function
import ROOT
from gna.ui import basecmd
from packages.parameters.lib.parameter_loader import get_parameters
from collections import OrderedDict
import warnings

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='Parameters group name')
        parser.add_argument('pars', nargs='*', help='Parameters to store')

        choices = ['free', 'constrained', 'fixed']
        parser.add_argument('-m', '--modes', nargs='+', default=['free', 'constrained'], choices=choices, help='Parameters to take')
        parser.add_argument('-v', '--verbose', action='count', help='verbose mode')

    def init(self):
        if self.opts.verbose>1:
            print('Loading parameter group {}:'.format(self.opts.name), ', '.join(self.opts.modes))

        self.loaded_parameters = OrderedDict((par.qualifiedName(), par) for par
                                             in get_parameters(self.opts.pars,
                                                 drop_fixed       = 'fixed' not in self.opts.modes,
                                                 drop_free        = 'free'  not in self.opts.modes,
                                                 drop_constrained = 'constrained' not in self.opts.modes,
                                                 )
                                             )

        self.env.future[('parameter_groups', self.opts.name)] = self.loaded_parameters

        if self.opts.verbose:
            print('Loaded parameters group {}, count {}: '.format(self.opts.name, len(self.loaded_parameters)))
        if self.opts.verbose>1:
            print(list(self.loaded_parameters.keys()))
