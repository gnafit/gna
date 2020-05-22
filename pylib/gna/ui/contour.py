# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from gna.ui import basecmd, set_typed

import argparse
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig

from matplotlib.ticker import AutoMinorLocator

import re
from itertools import product

try:
    import numpy as np
except ImportError as er:
    raise Exception("Numpy can't be imported")
try:
    import scipy.stats
    from scipy.interpolate import interp1d
    from scipy.optimize import bisect
    from scipy.special import erf
    from scipy.interpolate import griddata
except ImportError as er:
    raise Exception("Scipy module can't be imported")
try:
    from gna.pointtree import PointTree
except ImportError as er:
    raise Exception("PoinTree can't be imported")



class vmaparray(np.ndarray):
    def __new__(cls, arr, grids):
        obj = np.asarray(arr).view(cls)
        obj.grids = np.array(grids)
        obj.paths = ['/'.join(str(x) for x in values) for values in product(*grids)]
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grids = getattr(obj, 'grids', None)
        self.paths = getattr(obj, 'paths', None)

    def gridindex(self, values):
        return tuple(np.searchsorted(g, v) for g, v in zip(self.grids, values))

    def bypath(self, path):
        return self.ravel()[self.paths.index(path)]

def pvalue_nsigma( nsigma ):
    return 1 - scipy.stats.norm.cdf(-nsigma)*2

def chi2_nsigma( nsigma, ndf ):
    return scipy.stats.chi2.ppf( scipy.stats.chi2.cdf( nsigma**2, 1), ndf )

def gausspval(asimov, data):
    return 1.0-0.5*(1+erf(-(asimov-data)/np.sqrt(8*abs(asimov))))

def readchi2(grp, obs):
    return grp['datafit'][0]

def estimatepvalue(grp, obs):
    names = list(grp['dchi2s'].keys())
    assert len(names) == 1
    dist = grp['dchi2s'][names[0]][:]
    # dist = dist[dist >= 0]
    idx = dist.searchsorted(obs)
    if idx == 0:
        msg = "WARNING: statistic={} is smaller than any sample (min={})!"
        print(msg.format(obs, dist[0]))
        return 1
    if idx == len(dist):
        msg = "WARNING: statistic={} is larger than any sample (max={})!"
        print(msg.format(obs, dist[-1]))
        return 0
    p, n = dist[idx-1:idx+1]
    z = idx-1 + (obs-p)/(n-p)
    return 1.0 - z/len(dist)

vmaptypes = {
    'chi2': readchi2,
    'fc': estimatepvalue,
    'fcupper': estimatepvalue,
    'h0asimov': readchi2,
    'h1asimov': readchi2,
    'chi2minplt': readchi2,
    'h0': estimatepvalue,
    'h1': estimatepvalue,
}

def emptymap(grids):
    return vmaparray(np.full([len(g) for g in grids], np.nan, float), grids)

pvmaptypes = { }

def pvaluemap(mapname, *args, **kwargs):
    def setupmap(cls):
        pvmaptypes[mapname] = cls
        cls.name = mapname
        return cls
    return setupmap

class PValueMap(object):
    def __init__(self, base):
        self.base = base

    @property
    def ndim(self):
        return len(self.data.shape)

    def dchi2(self, statistic):
        chi2data = self.base.statistic(statistic)
        chi2map = self.base.readmap('chi2')
        dchi2 = chi2map - chi2data
        return dchi2

@pvaluemap("chi2ci")
class Chi2ConfidenceMap(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min')
        #  print 'dchi2', dchi2
        pvs = 1-scipy.stats.chi2.cdf(dchi2, len(dchi2.shape))
        #  print 'len', len(dchi2.shape)
        #  print 'pvs', pvs
        return vmaparray(pvs, dchi2.grids)

@pvaluemap("chi2minplt")
class Chi2MinMap(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min')
        return vmaparray(dchi2, dchi2.grids)

@pvaluemap("chi2upper")
class Chi2UpperLimitMap(Chi2ConfidenceMap):
    @property
    def data(self):
        vmap = super(Chi2UpperLimitMap, self).data
        return vmap/2

@pvaluemap("fc")
class FCMap(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min')
        pvmap = self.base.readmap('fc', dchi2)
        return pvmap

@pvaluemap("fcupper")
class FCUpperMap(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min')
        pvmap = self.base.readmap('fcupper', dchi2)
        return pvmap

@pvaluemap("clsgauss")
class CLsGaussianMap(PValueMap):
    @property
    def data(self):
        def gausspval(asimov, data):
            pvs = 1.0-0.5*(1+erf(-(asimov-data)/np.sqrt(8*abs(asimov))))
            return vmaparray(pvs, data.grids)

        dchi2data = self.dchi2('chi2min_null')

        chi2h01 = self.base.readmap('h0asimov')
        chi2h10 = self.base.readmap('h1asimov')

        dchi2h1 = -chi2h10
        dchi2h0 = chi2h01

        clh1 = gausspval(dchi2h1, dchi2data)
        clh0 = gausspval(dchi2h0, dchi2data)

        return clh1/clh0

@pvaluemap("cls")
class CLsMap(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min_null')
        pv1map = self.base.readmap('h1', dchi2)
        pv0map = self.base.readmap('h0', dchi2)
        return pv1map/pv0map

@pvaluemap("clh1")
class CLH1Map(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min_null')
        pv1map = self.base.readmap('h1', dchi2)
        return pv1map

@pvaluemap("clh0")
class CLH0Map(PValueMap):
    @property
    def data(self):
        dchi2 = self.dchi2('chi2min_null')
        pv1map = self.base.readmap('h0', dchi2)
        return pv1map

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-o', '--output', dest='output')
        parser.add_argument('-s', '--show', action='store_true')
        for maptype in vmaptypes:
            parser.add_argument('--'+maptype)
        parser.add_argument('--no-shift', action='store_true', default=False,
                            help='Shift the chi2 map by the value in data or not')
        parser.add_argument('--plot', dest='plots', nargs='+',
                            action='append', required=True)
        parser.add_argument('--points', dest='points', nargs='+',
                            action='append', required=False, default=[])
        parser.add_argument('--savepoints', dest='savepoints',
                            required=False)
        parser.add_argument('--xlog', action='store_true', help='Use log scale over x-axis')
        parser.add_argument('--drawgrid', '--grid', action='store_true')
        parser.add_argument('--dm32', action='store_true')
        parser.add_argument('--ylim', type=float, nargs=2)
        parser.add_argument('--xlim', type=float, nargs=2)
        parser.add_argument('--labels', nargs='+')
        parser.add_argument('--legend', nargs='+', help='legend position')
        parser.add_argument('--title', default='', required=False,
                            help='Set a title to the plot')
        parser.add_argument('--no-bestfit', action='store_false',
        help='Do not show best fit point with contour')
        parser.add_argument('--minimizer', action=set_typed(env.parts.minimizer))
        parser.add_argument('--minimizer-chi2', action=set_typed(env.parts.minimizer))
        parser.add_argument('--xlabel', default='', required=False)
        parser.add_argument('--ylabel', default='', required=False)
        parser.add_argument('--figure', action='store_true', help='Create new figure')

    def statistic(self, name):
        if self.opts.no_shift:
            return 0.0

        #  print name
        if name in self.statistics:
            return self.statistics[name]
        if name == 'chi2min':
            minimizer = self.opts.minimizer
        elif name == 'chi2min_null':
            minimizer = self.opts.minimizer_chi2
        else:
            raise Exception("unknown statistic", name)
        res = minimizer.fit()
        minimizer.PrintResults()
        assert res.success
        self.statistics[name] = res.fun
        # print minimizer.pars, res.x
        self.statpoints[name] = dict(zip(minimizer.pars, res.x))
        return res.fun

    def opentree(self, mapname):
        fname = getattr(self.opts, mapname)
        if not fname:
            msg = "required tree '{}' is not provided"
            raise Exception(msg.format(mapname))
        tree = PointTree(self.env, fname)
        if self.params is not None:
            assert list(self.params) == list(tree.params)
        else:
            self.params = tree.params
        return tree

    def readmap(self, mapname, obs=None):
        func = vmaptypes[mapname]
        bypath = hasattr(obs, 'bypath')
        points, vals = [], []
        with self.opentree(mapname) as tree:
            grids = tree.grids
            #  print 'grids', grids
            vmap = emptymap(grids)
            for path, values, grp in tree.iterall():
                points.append(values)
                if bypath:
                    vals.append(func(grp, obs.bypath(path)))
                else:
                    vals.append(func(grp, obs))
        mesh = np.meshgrid(*grids)

        if len(grids) == 1:
            interp = griddata(np.array(points), np.array(vals), np.array(mesh[0]), rescale=True, method='linear')
            vmap[...] = interp
        elif len(grids) == 2:
            interp = griddata(np.array(points), np.array(vals), (mesh[0], mesh[1]), rescale=True, method='linear')
            vmap[...] = interp.T
        return vmap

    def parselevel(self, pvspec):
        try:
            n, t = re.match('([0-9.]+)(.*)', pvspec).groups()
        except AttributeError:
            msg = "invalid CL specifier {}".format(pvspec)
            raise Exception(msg)
        n = float(n)
        if t in 'cl':
            if 50 < n < 100:
                pv = 1 - n/100
            elif 0 < n < 1:
                pv = 1 - n
            else:
                msg = "invalid value for CL: {}".format(n)
                raise Exception(msg)
        elif t == 'pv':
            if 50 < n < 100:
                pv = 1 - n/100
            elif 0 < n < 1:
                pv = 1 - n
            else:
                msg = "invalid value for p-value: {}".format(n)
                raise Exception(msg)
        elif t in ('s', 'sigma'):
            pv = 1 - scipy.stats.chi2.cdf(n**2, 1)
        else:
            msg = "invalid label CL specifier: {}".format(t)
            raise Exception(msg)
        return pv

    def run(self):
        self.params = None
        self.statistics = {}
        self.statpoints = {}
        self.ndim = None
        self.labels = None
        if self.opts.labels:
            self.labels = iter(self.opts.labels)

        minorLocatorx = AutoMinorLocator()
        minorLocatory = AutoMinorLocator()
        if self.opts.figure:
            plt.figure()
            ax = plt.subplot( 111 )
        else:
            ax = plt.gca()

        levels = set()
        colors = iter('bgrcmyk')
        for plotdesc in self.opts.plots:
            plottype, rest = plotdesc[0], plotdesc[1:]
            pvmap = pvmaptypes[plottype](self)
            color = self.plotmap(ax, pvmap)
            if color is None:
                color = next(colors)
            for pvspec in rest:
                pvlevel = self.parselevel(pvspec)
                #  print 'pvmap', pvmap.data
                #  print 'pvlevel', pvlevel
                self.plotlevel(ax, pvmap, pvlevel, color=color, pvspec=pvspec)
                levels.add(pvlevel)
        pointsfile = None
        if self.opts.savepoints:
            pointsfile = PointTree(self.env, self.opts.savepoints, "w")
            pointsfile.params = self.params
        for plotdesc in self.opts.points:
            plottype, pvspec, rest = plotdesc[0], plotdesc[1], plotdesc[2:]
            pvmap = pvmaptypes[plottype](self)
            if len(rest) > 1:
                width = (float(rest[0]), float(rest[1]))
            else:
                width = (0.05, 0.05)
            pvlevel = self.parselevel(pvspec)
            points = self.filterpoints(pvmap, pvlevel, width)[0]
            if pointsfile:
                for path in points[0]:
                    pointsfile.touch(path)
            self.plotpoints(ax, points[0])
        if self.ndim == 1:
            # plt.yticks(list(plt.yticks()[0]) + list(levels))
            # labels = ['%g' % loc for loc in plt.yticks()[0]]
            # ticks, labels = plt.yticks(plt.yticks()[0], labels)
            #plt.yticks(np.arange(self.opts.ylim[0], self.opts.ylim[1], (self.opts.ylim[1]-self.opts.ylim[0])/10.0))
            for level in levels:
                plt.axhline(level, linestyle='-.')
            xlabel = r'{}'.format(self.opts.xlabel)
            ylabel = r'{}'.format(self.opts.ylabel)
            ax.set_xlabel(xlabel, fontsize='xx-large')
            ax.set_ylabel(ylabel, fontsize='xx-large')
            if self.opts.xlog:
                ax.semilogx()
            ax.grid(False)
        elif self.ndim == 2 and self.opts.no_bestfit:
            if self.statpoints.get('chi2min'):
               xy = [self.statpoints['chi2min'][par] for par in
                       reversed(self.opts.minimizer.pars)]
               #  self.fixaxes(xy)
               print('bestfit', xy[0], xy[1])
               ax.plot(xy[0], xy[1], 'o', label='Best fit')

            #  ax.set_xlabel(r'$\sigma_{rel}$', fontsize='xx-large')
            if 'dm31' in self.params:
                if self.opts.dm32:
                    ax.set_ylabel(r'$\Delta m^2_{23}$', fontsize='xx-large')
                else:
                    ax.set_ylabel(r'$\Delta m^2_{13}$', fontsize='xx-large')
            elif 'theta13' in self.params:
                ax.set_ylabel(r'$\sin^2 2\theta_{13}$', fontsize='xx-large')
            ax.grid(True)
        if self.opts.ylim is not None:
            ax.set_ylim(self.opts.ylim)
        if self.opts.xlim is not None:
            ax.set_xlim(self.opts.xlim)
        if self.opts.drawgrid:
            ax.xaxis.set_minor_locator(minorLocatorx)
            ax.yaxis.set_minor_locator(minorLocatory)
            plt.tick_params(which='both', width=1)
            plt.tick_params(which='major', length=7)
            plt.tick_params(which='minor', length=4, color='k')
            ax.grid(which = 'minor', alpha = 0.3)
            ax.grid(which = 'major', alpha = 0.7)
        if self.opts.legend:
            plt.legend(loc=' '.join(self.opts.legend))
        if self.opts.title:
            plt.title(r'{0}'.format(self.opts.title))
        if self.opts.output:
            savefig(self.opts.output)
        if self.opts.show:
            plt.show()
        return True

    def fixaxes(self, axes):
        for i, param in enumerate(reversed(self.params)):
            if param == 'dm31' and self.opts.dm32:
                dm21 = self.localenv.exp.getPar("dm21").GetValue()
                print("dm21", dm21)
                axes[i] -= dm21

    def plotmap(self, ax, pvmap):
        ndim = pvmap.ndim
        if self.ndim is None:
            self.ndim = ndim
        else:
            assert self.ndim == ndim
        if ndim == 1:
            X = pvmap.data.grids[0]
            if self.labels:
                label = next(self.labels)
            else:
                label = pvmap.name
            lines = ax.plot(X, pvmap.data, label=label)
            return lines[0].get_color()
        elif ndim == 2:
            pass

    def plotlevel(self, ax, pvmap, pvlevel, color, pvspec):
        ndim = pvmap.ndim
        if ndim == 1:
            X = pvmap.data.grids[0]
            f = interp1d(X, pvmap.data)
            try:
                x = bisect(lambda x: f(x)-pvlevel, X[0], X[-1])
            except ValueError:
                return
            ax.axvline(x, color=color, linewidth=1, linestyle='--')
        elif ndim == 2:
            XX, YY = np.meshgrid(*reversed(pvmap.data.grids))
            self.fixaxes([XX, YY])
            CS = ax.contour(XX, YY, pvmap.data, levels=(pvlevel,))
            for c in CS.collections:
                c.set_color(color)
                if self.labels:
                    c.set_label(next(self.labels))
                else:
                    c.set_label("{} {}".format(pvmap.name, pvspec))

    def plotpoints(self, ax, pts):
        points = np.array([[float(x) for x in path.split('/')] for path in pts])
        xy = [points[:, 1], points[:, 0]]
        ax.plot(xy[0], xy[1], 'o')
        return xy

    def filterpoints(self, pvmap, level, width):
        vmap = pvmap.data
        naxis = len(vmap.grids)

        def extend_map(vmap):
            evmap = np.zeros(shape=list(i+2 for i in vmap.shape), dtype=vmap.dtype)
            evmap[...] = np.nan
            centralblock = tuple(slice(1, i+1) for i in vmap.shape)
            evmap[centralblock] = vmap
            return evmap

        def extend_grid(grid):
            assert len(grid.shape) == 1
            egrid = np.zeros(shape=(grid.shape[0]+2,), dtype=grid.dtype)
            egrid[1:len(grid)+1] = grid
            egrid[0] = grid[0] + (grid[0] - grid[1])/2
            egrid[-1] = grid[-1] + (grid[-1] - grid[-2])/2
            return egrid

        def projection(axis, cell):
            proj = [c+1 for c in cell]
            proj[axis] = slice(cell[axis], cell[axis]+3)
            return tuple(proj)

        def slices(cell):
            return tuple([slice(c+1, c+3) for c in cell])

        def eslices(cell):
            return tuple([slice(c, c+3) for c in cell])

        def getpath(values):
            return '/'.join(str(x) for x in values)

        def getvalues(idx):
            return np.array([grid[j] for grid, j in zip(vmap.grids, idx)])

        diff = vmap/level - 1

        evmap = extend_map(vmap)
        ediff = evmap/level - 1

        egrids = [extend_grid(grid) for grid in vmap.grids]
        grad = np.zeros(shape=vmap.shape+(naxis,))
        grad[...] = np.nan

        idxes = set()
        newpaths = set()

        grads = []
        oldpaths = []

        for cell in np.ndindex(*vmap.shape):
            for axis in range(naxis):
                p = projection(axis, cell)
                rawstencil = ediff[p]
                mask = np.isfinite(rawstencil)
                if np.count_nonzero(mask) < 2:
                    grad[cell][axis] = np.nan
                    continue
                stencil = rawstencil[mask]
                dx = egrids[axis][p[axis]][mask]
                grad[cell][axis] = (stencil[-1]-stencil[0])/(dx[-1]-dx[0])

        splits = np.zeros(shape=evmap.shape+(naxis,), dtype='i')
        for cell in np.ndindex(*vmap.shape):
            window = ediff[eslices(cell)]
            if np.isnan(window).all():
                continue
            containing = (window >= 0).any() and (window <= 0).any()
            closeless = np.logical_and(window <= 0, abs(window) <= width[0])
            closemore = np.logical_and(window >= 0, abs(window) <= width[1])
            close = np.logical_or(closeless, closemore).any()
            if not (containing or close):
                continue

            values = getvalues(cell)
            oldpaths.append(getpath(values))
            der = grad[cell]
            grads.append(der)

            ecell = tuple([x+1 for x in cell])
            for axis in range(naxis):
                necell = tuple([x+int(i == axis) for i, x in enumerate(ecell)])
                cv = diff[cell]
                nv = ediff[necell]
                if np.isnan(cv) or np.isnan(nv):
                    continue
                if cv < -width[0] and nv < -width[0] or cv > width[1] and nv > width[1]:
                    continue
                #if abs(abs(cv)-abs(nv)) < (1-level)/10:
                    #continue
                splits[ecell][...] = 1
            print(abs(window))
            for axis in range(naxis):
                p = projection(axis, cell)
                dx = egrids[axis][p[axis]]
                step = (dx[2] - dx[1])/(splits[ecell][axis]+1)
                for r in range(splits[ecell][axis]):
                    newvalues = values.copy()
                    newvalues[axis] += (r+1)*step
                    newpaths.add(getpath(newvalues))
                    for offs in product(*[(-1, 0, 1)]*naxis):
                        newvalues2 = newvalues.copy()
                        for offaxis, off in enumerate(offs):
                            if off == 0:
                                continue
                            if offaxis == axis:
                                newvalues2[axis]+=off*step
                            else:
                                p = projection(offaxis, cell)
                                dx = egrids[offaxis][p[offaxis]]
                                if off == -1:
                                    if ecell[offaxis] == 1:
                                        break
                                    step2 = (dx[1] - dx[0])/(splits[ecell][offaxis]+1)
                                else:
                                    step2 = (dx[2] - dx[1])/(splits[ecell][offaxis]+1)
                                newvalues2[offaxis]+=off*step2
                        else:
                            newpaths.add(getpath(newvalues2))
        return (oldpaths, np.array(grads)), newpaths
