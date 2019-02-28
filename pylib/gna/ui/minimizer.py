"""Minimizer module: initializes minimizer for a given statistic and set of parameters"""

import ROOT
from gna.ui import basecmd, set_typed
from gna.minimizers import minimizers
from gna.minimizers import spec
from gna.parameters.parameter_loader import get_parameters
from gna.config import cfg
import warnings

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-s', '--spec', default=None, help='Minimization options (YAML)')
        parser.add_argument('name', help='Minimizer name')
        parser.add_argument('type', choices=minimizers.keys())
        parser.add_argument('statistic', action=set_typed(env.parts.statistic))
        parser.add_argument('par', nargs='*')

    def init(self):
        minimizer = minimizers[self.opts.type](self.opts.statistic)

        loaded_parameters = get_parameters(self.opts.par)
        statistic_parameters = []
        for par in loaded_parameters:
            if par.influences(self.opts.statistic):
                statistic_parameters.append(par)
            elif cfg.debug_par_fetching:
                warnings.warn("parameter {} doesn't influence the statistic and is being dropped".format(par.name()))
            else:
                continue

        minimizer.addpars(statistic_parameters)

        if self.opts.spec is not None:
            minimizer.spec = spec.parse(self.env, minimizer, self.opts.spec)

        self.env.parts.minimizer[self.opts.name] = minimizer
