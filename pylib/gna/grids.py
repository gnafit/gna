import argparse
from collections import OrderedDict
from itertools import product, islice, chain, repeat

import numpy as np
import ROOT

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
                param, start, count, step = values
                griddesc = (start, self.getcount(count), self.getstep(step))
            elif gridtype == 'log':
                param, start, end, count = values
                griddesc = (start, end, self.getcount(count))
            elif gridtype == 'lin':
                param, start, end, count = values
                griddesc = (start, end, self.getcount(count))
            elif gridtype == 'list':
                param = values[0]
                griddesc = values[1:]
            namespace.grids.append((env.pars[param], gridtype, griddesc))
    return GridAction

class PointSet(object):
    def __init__(self, opts, params, pointsfactory):
        self.opts = opts
        self.params = params
        self.pointsfactory = pointsfactory

    @classmethod
    def fromgrids(cls, opts):
        gridparams = OrderedDict()
        for par, gridtype, griddesc in opts.grids:
            if not isinstance(par, ROOT.GaussianParameter("double")):
                raise TypeError("{} is not independent parameter, nop possible to scan over it".format(par.name()))
            gridparams[par] = None
        params = list(gridparams.keys())
        return cls(opts, params, lambda: product(*cls.decomposed(opts)))

    @classmethod
    def fromtree(cls, opts, tree):
        if opts.grids:
            msg = 'grids specified with point tree'
            raise Exception(msg)
        params = tree.params
        return cls(opts, params, tree.itervalues)

    @classmethod
    def addargs(cls, parser, env):
        parser.add_argument('--grid', dest='grids', action=gridaction('range', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "COUNT", "STEP"))
        parser.add_argument('--loggrid', dest='grids', action=gridaction('log', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "COUNT"))
        parser.add_argument('--lingrid', dest='grids', action=gridaction('lin', env), default=[],
                            nargs=4, metavar=("PARAM", "START", "END", "COUNT"))
        parser.add_argument('--listgrid', dest='grids', action=gridaction('list', env), default=[],
                            nargs='+', metavar=("PARAM", "VALUE"))
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--pointsrange', type=int, nargs='+', default=None)
        group.add_argument('--pointspaths', nargs='+', default=None)

    @classmethod
    def decomposed(cls, opts):
        cast = float
        grids = OrderedDict()
        for pname, gridtype, griddesc in opts.grids:
            start = griddesc[0]
            startval = cast(start)
            if gridtype == 'range':
                count, step = griddesc[1:]
                grid = np.arange(startval, startval+count*step, step)[:count]
            elif gridtype == 'log':
                end, count = griddesc[1:]
                endval = cast(end)
                logstart, logend = np.log10([startval, endval])
                grid = 10**np.linspace(logstart, logend, count)
            elif gridtype == 'lin':
                end, count = griddesc[1:]
                endval = cast(end)
                grid = np.linspace(startval, endval, count)
            elif gridtype == 'list':
                grid = np.array([cast(x) for x in griddesc])
            else:
                message = "unknown grid type `{}'".format(gridtype)
                raise ValueError(message)
            grids[pname] = np.unique(np.hstack([grids.get(pname, []), grid]))
        return list(grids.values())

    def iterpathvalues(self, ngrids=None, grid_idx=None):
        it = self.pointsfactory()
        if self.opts.pointspaths:
            for path in self.opts.pointspaths:
                path = path.strip('/')
                yield path, [p.cast(x) for p, x in zip(self.params, [_f for _f in path.rsplit('/') if _f])]
            return
        if self.opts.pointsrange:
            assert(len(self.opts.pointsrange) <= 3)
            sliceargs = islice(chain(self.opts.pointsrange, repeat(None)), 3)
            it = islice(it, *sliceargs)

        if (ngrids is not None) and (grid_idx is not None):
           total = np.array_split(list(it), ngrids)
           it = iter(total[grid_idx])
        for values in it:
            yield ('/'.join([str(x) for x in values]), values)
