# -*- coding: utf-8 -*-
"""Minimizer module: initializes minimizer for a given statistic and set of parameters"""

from __future__ import absolute_import
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
        parser.add_argument('type', choices=minimizers.keys(),
                                    help='Minimizer type {%(choices)s}', metavar='minimizer')
        parser.add_argument('statistic', action=set_typed(env.parts.statistic), help='Statistic name',
                                         metavar='statmodule')
        parser.add_argument('par', nargs='*', help='Parameters to minimize')
        parser.add_argument('--drop-constrained', action='store_true', help='drop constrained arguments')

    def init(self):
        statistic = ROOT.StatisticOutput(self.opts.statistic.transformations.back().outputs.back())
        minimizer = minimizers[self.opts.type](statistic)

        loaded_parameters = get_parameters(self.opts.par, drop_fixed=True, drop_free=False, drop_constrained=self.opts.drop_constrained)
        statistic_parameters = []
        for par in loaded_parameters:
            if par.influences(self.opts.statistic.transformations.back()):
                statistic_parameters.append(par)
            elif cfg.debug_par_fetching:
                warnings.warn("parameter {} doesn't influence the statistic and is being dropped".format(par.name()))
            else:
                continue

        minimizer.addpars(statistic_parameters)

        if self.opts.spec is not None:
            minimizer.spec = spec.parse(self.env, minimizer, self.opts.spec)

        self.env.parts.minimizer[self.opts.name] = minimizer
