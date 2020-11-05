"""A group of paraeters UI: select a set of parameters and store"""

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

        parser.add_argument('-n', '--name', dest='name', help='Parameters group name')
        parser.add_argument('-p', '--pars', dest='pars', nargs='*', help='Parameters to store')

        parser.add_argument('-v', '--verbose', action='count', help='verbose mode')

        filter1 = parser.add_argument_group(title='Filter', description='Arguments to filter the list of parameters')
        choices = ['free', 'constrained', 'fixed']
        filter1.add_argument('-m', '--modes', nargs='+', default=['free', 'constrained'], choices=choices, help='Parameters to take')

        filter1.add_argument('-x', '--exclude', nargs='*', default=[], help='parameters to exclude (pattern in fullname)')
        filter1.add_argument('-i', '--include', nargs='*', default=None, help='parameters to include, exclusive (pattern in fullname)')

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
        self.loaded_parameters = OrderedDict(filter(self._keep_parameter, self.loaded_parameters.items()))

        self.env.future[('parameter_groups', self.opts.name)] = self.loaded_parameters

        if self.opts.verbose:
            print('Loaded parameters group {}, count {}: '.format(self.opts.name, len(self.loaded_parameters)))
        if self.opts.verbose>1:
            print(list(self.loaded_parameters.keys()))

    def _keep_parameter(self, (name, par)):
        for excl in self.opts.exclude:
            if excl in name:
                return False

        include = self.opts.include
        if include is None:
            return True

        for incl in include:
            if incl in name:
                return True

        return False



