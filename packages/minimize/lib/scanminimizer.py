#!/usr/bin/env python

from typing import List, Any, Mapping
from collections import abc
from copy import copy

import ROOT
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # do basic counting if tqdm is missing
    def counting_mock(iterable, total):
        for i, elem in enumerate(iterable):
            print(f"{i}/{total}")
            yield elem
    tqdm = counting_mock

from minimize.lib.base import MinimizerBase, FitResult


class ScanMinimizer(MinimizerBase):
    _label           = 'ScanMinimizer'
    _results         = None
    _result_min      = None
    _result_improved = None
    def __init__(self, statistic, minpars, gridpars, extraminimizerclass,
            extraminimizeropts={}, fixed_order=[], **kwargs):
        self.verbose = kwargs.pop('verbose', False)
        MinimizerBase.__init__(self, statistic, minpars, **kwargs)
        self._extraminimizer = extraminimizerclass(statistic, minpars, **extraminimizeropts)
        self._grid = ParsGrid(minpars, gridpars)
        self.fixed_order = fixed_order

    @property
    def label(self):
        return '+'.join((self._label, self._extraminimizer.label))

    @property
    def results(self):
        return self._results

    @property
    def result_min(self):
        return self._result_min

    @property
    def result_improved(self):
        return self._result_improved

    def setuppars(self):
        if not self.parspecs.modified:
            return

        self.update_minimizable()
        self.parspecs.resetstatus()

    def grid_scan(self, verbose:bool=False):
        '''Return test statistic values in all points of the grid'''
        self.setuppars()
        self.update_minimizable()
        grid_it = self._grid.progress_it() if self.verbose else iter(self._grid)
        return [self._extraminimizer.fit() for it in grid_it ]

    def _scan(self):
        '''Returns a 5-tuple:
           - results of fit,
           - test statistic value in each point,
           - mask for successful fits,
           - values of parameters on grid,
           - names of scanned parameters'''
        results = ()
        funs = ()
        mask = ()
        par_values = ()
        grid_it = self._grid.progress_it() if self.verbose else iter(self._grid)
        for i, step in enumerate(grid_it):
            res = self._extraminimizer.fit()
            results += res,
            funs += res['fun'],
            mask += not res['success'],
            par_values += tuple((par.value for par in step))
        par_names = tuple(par.name for par in step)

        return results, np.array(funs), np.array(mask), par_values, par_names

    def _child_fit(self, **kwargs):
        self.setuppars()
        self.update_minimizable()
        with self.parspecs:
            with FitResult() as fr:
                self._results, funs, mask, *_ = self._scan()

                funs = np.ma.array(funs, mask=mask)
                if mask.any():
                    funs[mask]=1.e100
                if (~funs.mask).any():
                    imin = np.argmin(funs)
                else:
                    imin=0
                self._result_min = self._results[imin]

                for parname in self._result_min['fixed']:
                    val = self._result_min['xdict'][parname]
                    self._parspecs[parname].value=val

                self._result_improved = self._extraminimizer.fit(**kwargs)

        nfev = sum(r['nfev'] for r in self._results) + self._result_improved['nfev']
        fr.set(x=self._result_improved['x'], errors=self._result_improved['errors'], fun=self._result_improved['fun'],
               success=self._result_improved['success'], message=self._result_improved['message'],
               minimizer=self.label, nfev=nfev,
               hess_inv = self._result_improved.get('hess_inv'),
               jac = self._result_improved.get('jac'),
               errors_profile = self._result_improved.get('errors_profile', {}),
               covariance = self._result_improved.get('covariance', {})
               )
        self._result = fr.result
        self.patchresult()

        return self._result

    def saveresult(self, loc):
        storage = loc.child(self.name)
        storage['results'] = self._results
        storage['result_min'] = self._result_min
        storage['result_improved'] = self._result_improved

class ParOnGrid:
    def __init__(self, minpar, grid):
        minpar.scanvalues = grid
        self.minpar = minpar

    def __iter__(self):
        fixed_save = self.minpar.fixed
        value_save = self.minpar.value
        self.minpar.fixed = True

        for v in self.minpar.scanvalues:
            self.minpar.value = v
            yield self.minpar

        self.minpar.fixed = fixed_save
        self.minpar.value = value_save

class ParsGrid:
    def __init__(self, minpars, gridpars):
        self._minpars = minpars
        self._gridpars = gridpars
        self._grid = []

        self.checkpars()

    def checkpars(self):
        if not self._gridpars:
            return
        for key, grid in self._gridpars.walkdicts():
            try:
                # Replace parameter with minimizer parameter specification
                minpar = self._minpars[grid['par']]
                if not any(minpar is par.minpar for par in self._grid):
                    self._grid.append(ParOnGrid(minpar, grid['grid']))
            except KeyError:
                raise Exception('Parameter {} is not in minimizable parameters'.format('.'.join(key)))

    def __iter__(self):
        yield from product_it(*self._grid)

    def progress_it(self):
        '''Iterator with tqdm progress bar for easier track of progress during
        scanning'''
        # count total number of points in grid
        total = np.prod([len(par.minpar.scanvalues) for par in self._grid])
        yield from tqdm(product_it(*self._grid), total=total)


def product_it(*args):
    n = len(args)
    pos = 0
    iterators = [None]*n
    ret = [None,]*n

    it_pos=0

    while True:
        if it_pos<n and not iterators[it_pos]:
            iterators[it_pos]=iter(args[it_pos])
            it_pos+=1
            continue

        try:
            ret[pos] = next(iterators[pos])
            if pos==n-1:
                yield tuple(ret)
                continue
            pos+=1
            continue
        except StopIteration:
            it_pos=pos
            iterators[it_pos]=None
            pos-=1
            if pos==-1:
                break
            continue
