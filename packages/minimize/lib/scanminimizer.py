#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from scipy.optimize import minimize
from packages.minimize.lib.base import MinimizerBase, FitResult
import ROOT
import numpy as np

class ScanMinimizer(MinimizerBase):
    _label           = 'ScanMinimizer'
    _results         = None
    _result_min      = None
    _result_improved = None
    def __init__(self, statistic, minpars, gridpars, extraminimizerclass, extraminimizeropts={}, **kwargs):
        MinimizerBase.__init__(self, statistic, minpars, **kwargs)
        self._extraminimizer = extraminimizerclass(statistic, minpars, **extraminimizeropts)
        self._grid = ParsGrid(minpars, gridpars)

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

    def _scan(self):
        results = ()
        funs = ()
        mask = ()
        for i, step in enumerate(self._grid):
            res = self._extraminimizer.fit()
            results += res,
            funs += res['fun'],
            mask += not res['success'],

        return results, funs, mask

    def _child_fit(self, profile_errors=None):
        self.setuppars()
        self.update_minimizable()
        with self.parspecs:
            with FitResult() as fr:
                self._results, funs, mask = self._scan()

                funs = np.ma.array(funs, mask=mask)
                imin = np.argmin(funs)
                self._result_min = self._results[imin]

                for parname in self._result_min['fixed']:
                    val = self._result_min['xdict'][parname]
                    self._parspecs[parname].value=val

                self._result_improved = self._extraminimizer.fit(profile_errors=profile_errors)

        nfev = sum(r['nfev'] for r in self._results) + self._result_improved['nfev']
        fr.set(x=self._result_improved['x'], errors=self._result_improved['errors'], fun=self._result_improved['fun'],
               success=self._result_improved['success'], message=self._result_improved['message'],
               minimizer=self.label, nfev=nfev,
               hess_inv = self._result_improved.get('hess_inv'),
               jac = self._result_improved.get('jac')
               )
        self._result = fr.result
        self.patchresult()

        return self._result

    def saveresult(self, loc):
        storage = loc.child(self.name)
        storage['results'] = self._results
        storage['result_min'] = self._result_min
        storage['result_improved'] = self._result_improved
        import IPython; IPython.embed()

class ParOnGrid(object):
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

class ParsGrid(object):
    def __init__(self, minpars, gridpars):
        self._minpars = minpars
        self._gridpars = gridpars
        self._grid = []

        self.checkpars()

    def checkpars(self):
        for key, grid in self._gridpars.walkdicts():
            try:
                # Replace parameter with minimizer parameter specification
                minpar = self._minpars[grid['par']]
                self._grid.append(ParOnGrid(minpar, grid['grid']))
            except KeyError:
                raise Exception('Parameter {} is not in minimizable parameters'.format('.'.join(key)))

    def __iter__(self):
        for step in product_it(*self._grid):
            yield step

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
