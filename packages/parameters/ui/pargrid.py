"""Specify a grid for a few parameters to be used with scanning minimizer."""

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
                            nargs=4, metavar=("PARAM", "STARTPOWER", "ENDPOWER", "COUNT"),
                            help='log grid (np.logspace)')
        parser.add_argument('--geomspace', dest='grids', action=gridaction('geomspace', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "COUNT"),
                            help='log grid (np.geomspace)')
        parser.add_argument('--linspace', dest='grids', action=gridaction('linspace', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "COUNT"),
                            help='linear grid (np.linspace)')
        parser.add_argument('--list', dest='grids', action=gridaction('list', env), default=[],
                            nargs='+', metavar=("PARAM", "VALUE"),
                            help='grid, specified via a list of values')
        parser.add_argument('-v', '--verbose', action='count', help='be more verbose')

    def init(self):
        storage = self.env.future.child(('pargrid', self.opts.name.split('.')))

        if self.opts.verbose:
            print('Provide grids:')
        for parname, par, grid, gridtype in self.opts.grids:
            storage[parname] = dict(par=par, grid=grid)

            if self.opts.verbose:
                print('  {}: {} of {} from {} to {} (incl)'.format(parname, gridtype, grid.size, grid[0], grid[-1]))
                if self.opts.verbose>1:
                    print('  {}: {!s}'.format(parname, grid))


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
            param, values = values[0], values[1:]
            if gridtype == 'range':
                start, end, step = values
                start, end = float(start), float(end)
                step = self.getstep(step)
                grid = np.arange(start, end, step, dtype='d')
            elif gridtype == 'geomspace':
                start, end, count = values
                start, end = float(start), float(end)
                count = self.getcount(count)
                grid = np.geomspace(start, end, count, dtype='d')
            elif gridtype == 'logspace':
                start, end, count = values
                start, end = float(start), float(end)
                count = self.getcount(count)
                grid = np.logspace(start, end, count, dtype='d')
            elif gridtype == 'linspace':
                start, end, count = values
                start, end = float(start), float(end)
                count = self.getcount(count)
                grid = np.linspace(start, end, count, dtype='d')
            elif gridtype == 'list':
                grid = np.array(values, dtype='d')
            par = env.pars[param]
            namespace.grids.append((param, par, grid, gridtype))
    return GridAction

cmd.__tldr__ = """\
                The module provides tools for creating grids to scan over parameters.
                It supports `range`, `linspace`, `logspace` and `geomspace` which are similar to their analogues from `numpy`.
                It also supports a list of values passed from the command line.

                \033[32mGenerate a linear grid for the parameter 'E0':
                ```sh
                ./gna -- \\ 
                    -- gaussianpeak --name peak \\ 
                    -- pargrid scangrid --linspace peak.E0 0.5 4.5 10 -vv
                ```

                The possible options include:
                | Option        | Arguments                      | NumPy analogue | Includes end point |
                |:--------------|:-------------------------------|:---------------|:-------------------|
                | `--range`     | `start` `stop` `step`          | arange         | ✘                  |
                | `--linspace`  | `start` `stop` `n`             | linspace       | ✔                  |
                | `--geomspace` | `start` `stop` `n`             | geomspace      | ✔                  |
                | `--logspace`  | `start_power` `stop_power` `n` | logspace       | ✔                  |
                | `--list`      | space separated values         | array          | ✔                  |

                \033[32mProvide a list of grid values from a command line:
                ```sh
                ./gna -- \\ 
                    -- gaussianpeak --name peak \\ 
                    -- pargrid scangrid --linspace peak.E0 1 2 8 -vv
                ```

                See also: `minimizer-scan`.
             """
