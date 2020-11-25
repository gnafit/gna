"""Select a group of parameters for the minimization and other purposes."""

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

        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbose mode')

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

    def _keep_parameter(self, namepar):
        name, par = namepar
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

    __tldr__ = """\
                The module recursively selects parameters based on their status (free, constrained, fixed)
                and inclusion/exclusion mask.
                The list is stored in `env.future` and may be used by minimizers.
                By default the module selects all the not fixed parameters: free and constrained.

                \033[32mSelect not fixed parameters from the namespace 'peak' and store as 'minpars':
                \033[31m./gna \\
                    -- gaussianpeak --name peak \\
                    -- ns --name peak --print \\
                          --set E0             values=2.5  free \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 fixed \\
                    -- pargroup minpars peak -vv\033[0m

                The '-m' option may be used with few arguments describing the parameter mode. The choices include:
                free, constrained and fixed.

                \033[32mSelect only _fixed_ parameters from the namespace 'peak' and store as 'minpars':
                \033[31m./gna \\
                    -- gaussianpeak --name peak \\
                    -- ns --name peak --print \\
                          --set E0             values=2.5  free \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 fixed \\
                    -- pargroup minpars peak -m fixed -vv\033[0m

                The parameters may be filtered with '-x' and '-i' flags. The option '-x' will exclude parameters,
                full names of which contain one of the string passed as arguments. The option '-i' will include
                only matching parameters.

                See also: `minimizer-v1`, `minimizer-scan`
               """
