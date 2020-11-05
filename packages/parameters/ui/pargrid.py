"""Specify a grid for the parameters"""

import ROOT
from gna.ui import basecmd
from packages.parameters.lib.parameter_loader import get_parameters
from collections import OrderedDict
import numpy as np
import argparse
import warnings

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='Parameters group name')

        parser.add_argument('--range', dest='grids', action=gridaction('range', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "STEP"),
                            help='range grid (np.arange)')
        parser.add_argument('--logspace', dest='grids', action=gridaction('logspace', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "COUNT"),
                            help='linear grid (np.linespace)')
        parser.add_argument('--linspace', dest='grids', action=gridaction('linspace', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "COUNT"),
                            help='logspace grid (np.geomspace)')
        parser.add_argument('--list', dest='grids', action=gridaction('list', env), default=[],
                            nargs='+', metavar=("PARAM", "VALUE"),
                            help='grid, spericied via a list of values')

    def init(self):
        storage = self.env.future.child(('pargrid', self.opts.name.split('.')))

        for parname, par, grid in self.opts.grids:
            storage[parname] = dict(par=par, grid=grid)

class _AddGridActionBase(argparse.Action):
    def getcount(self, count):
        try:
            return int(count)
        except:
            message = "count should be int, `{}' is given".format(count)
            raise argparse.ArgumentError(self, message)
        if not count > 0:
            message = "count should be positive, `{}' is given".format(count)
            raise argparse.ArgumentError(self, message)

    def getstep(self, step):
        try:
            step = float(step)
        except:
            message = "step should be float, `{}' is given".format(step)
            raise argparse.ArgumentError(self, message)
        if step == 0.0:
            raise argparse.ArgumentError(self, "step should be nonzero")
        return step

def gridaction(gridtype, env):
    class GridAction(_AddGridActionBase):
        def __call__(self, parser, namespace, values, option_string=None):
            if gridtype == 'range':
                param, start, end, step = values
                start, end = float(start), float(end)
                step = self.getstep(step)
                grid = np.arange(start, end, step, dtype='d')
            elif gridtype == 'logspace':
                param, start, end, count = values
                start, end = float(start), float(end)
                count = self.getcount(count)
                grid = np.geomspace(start, end, count, dtype='d')
            elif gridtype == 'linspace':
                param, start, end, count = values
                start, end = float(start), float(end)
                count = self.getcount(count)
                grid = np.linspace(start, end, count, dtype='d')
            elif gridtype == 'list':
                param = values[0]
                grid = values[1:]
            par = env.pars[param]
            namespace.grids.append((param, par, grid))
    return GridAction

