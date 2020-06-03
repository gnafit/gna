#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from scipy.optimize import minimize
from packages.minimize.lib.base import MinimizerBase, FitResult
import ROOT
import numpy as np

class ScanMinimizer(MinimizerBase):
    _label       = 'scan-minimizer'
    def __init__(self, statistic, minpars, gridpars, extraminimizerclass, extraminimizeropts={}):
        MinimizerBase.__init__(self, statistic, minpars)
        self._extraminimizer = extraminimizerclass(statistic, minpars, **extraminimizeropts)
        self._gridpars = gridpars.deepcopy()

    @property
    def label(self):
        return '+'.join((self._label, self._extraminimizer.label))

    def checkpars(self):
        for key, grid in self._gridpars.walkdicts():
            if not grid['par'] in self.parspecs:
                raise Exception('Parameter {} is not in minimizable parameters'.format('.'.join(key)))

    def setuppars(self):
        self.checkpars()
        if not self.parspecs.modified:
            return

        self.update_minimizable()
        self.parspecs.resetstatus()

    def _scan(self):
        import IPython; IPython.embed()
        pass

    def _child_fit(self, profile_errors=None):
        self.setuppars()
        self.update_minimizable()
        with self.parspecs:
            with FitResult() as fr:
                self._scan()

        # fr.set(x=self._res.x, errors=None, fun=self._res.fun,
               # success=self._res.success, message=self._res.message,
               # minimizer=self.label, nfev=self._res.nit,
               # hess_inv = self._res.hess_inv,
               # jac = self._res.jac
                # )
        # self._result = fr.result
        # self.patchresult()

        return self.result


