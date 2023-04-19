from itertools import product
from functools import lru_cache
from typing import Union
from pathlib import Path
import atexit

import numpy as np
import scipy
import scipy.stats
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.optimize import bisect

from minimize.lib.pointtree_v01 import PointTree_v01
from gna.env import env

_trees = []

@atexit.register
def __close_trees():
    global trees
    for tree in _trees:
        tree.close()

class Chi2Reader:
    def __init__(self, fpath, reader_func=None):
        self.fpath: Union[str, Path] = fpath
        global _trees
        _trees.append(PointTree_v01(env, fpath))
        self.tree = _trees[-1]
        self.params = self.tree.params

    @property
    @lru_cache(maxsize=32)
    def chi2map(self):
        func = readchi2
        points, vals = [], []
        grids = self.tree.grids[::-1]
        vmap = emptymap(grids)
        for path, values, grp in self.tree.iterall():
            points.append(values[::-1])
            vals.append(func(grp))
        mesh = np.meshgrid(*reversed(grids))
        points = np.array(points)
        if points.shape[1] == 2:
            interp = griddata(points, np.array(vals), (mesh[0], mesh[1]),
                              fill_value=0., rescale=True, method='cubic')
        if np.array(points).shape[1] == 1:
            interp = self.interp1d(np.sort(points[:, 0]))
        vmap[...] = interp
        return vmap

    @property
    @lru_cache(maxsize=32)
    def interp1d(self):
        func = readchi2
        points, vals = [], []
        grids = self.tree.grids[::-1]
        for path, values, grp in self.tree.iterall():
            points.append(values[::-1])
            vals.append(func(grp))
        points = np.array(points)[:,0]
        idxes_sort = np.argsort(points)
        points = points[idxes_sort]
        vals = np.array(vals)[idxes_sort]
        return interp1d(points, vals, kind='cubic')

    @property
    @lru_cache(maxsize=32)
    def interp1d_log(self):
        func = readchi2
        points, vals = [], []
        grids = self.tree.grids[::-1]
        for path, values, grp in self.tree.iterall():
            points.append(values[::-1])
            vals.append(func(grp))
        points = np.array(points)[:,0]
        idxes_sort = np.argsort(points)
        points = points[idxes_sort]
        points = np.log10(points)
        vals = np.array(vals)[idxes_sort]
        return interp1d(points, vals, kind='cubic')


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

def readchi2(grp):
    return grp['datafit'][0]

def emptymap(grids):
    return vmaparray(np.full([len(g) for g in grids], np.nan, float), grids)

pvmaptypes = { }

def pvaluemap(mapname, *args, **kwargs):
    def setupmap(cls):
        pvmaptypes[mapname] = cls
        cls.name = mapname
        return cls
    return setupmap

class PValueMap:
    def __init__(self, reader: Chi2Reader, bestfit: float):
        self.chi2_map = reader.chi2map
        self.bestfit = bestfit

    @property
    @lru_cache(maxsize=32)
    def ndim(self):
        return len(self.data.shape)

    @property
    @lru_cache(maxsize=32)
    def dchi2(self):
        chi2data = self.bestfit
        chi2map = self.chi2_map
        dchi2 = chi2map - chi2data
        return dchi2

@pvaluemap("chi2ci")
class Chi2ConfidenceMap(PValueMap):
    @property
    @lru_cache(maxsize=32)
    def data(self):
        dchi2 = self.dchi2
        pvs = 1-scipy.stats.chi2.cdf(dchi2, len(dchi2.shape))
        return vmaparray(pvs, dchi2.grids[::-1])

@pvaluemap("chi2profile")
class Chi2MinMap(PValueMap):
    @property
    @lru_cache(maxsize=32)
    def data(self):
        dchi2 = self.dchi2
        return vmaparray(dchi2, dchi2.grids[::-1])
