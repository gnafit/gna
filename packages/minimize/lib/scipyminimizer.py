#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from scipy.optimize import minimize
from packages.minimize.lib.base import MinimizerBase, FitResult
import ROOT
import numpy as np

class SciPyMinimizer(MinimizerBase):
    _label       = 'scipy'
    _method      = 'BFGS'
    _kwargs      = None
    def __init__(self, statistic, minpars, method=None, **kwargs):
        MinimizerBase.__init__(self, statistic, minpars, **kwargs)
        if method:
            self._method = method

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def label(self):
        return '.'.join((self._label, self._method))

    def setuppars(self):
        if self.kwargs is not None and not self.parspecs.modified:
            return

        self.update_minimizable()

        npars = len(self.parspecs)
        self._kwargs = dict(
                x0       = [None]*npars,
                bounds   = [None]*npars,
                method   = self._method
                )

        bounded = False
        for i, (name, parspec) in enumerate(self.parspecs.items()):
            self._kwargs['x0'][i]     = parspec.value
            bounds                    = (parspec.vmin, parspec.vmax)
            self._kwargs['bounds'][i] =  bounds
            if parspec.fixed:
                raise Exception('Do not work with fixed parameters')

            if bounds!=(None, None):
                bounded=True

        if not bounded:
            del self._kwargs['bounds']

        self.parspecs.resetstatus()

    def call(self, args0):
        """Evaluate the function for a given set of values"""
        args = np.ascontiguousarray(args0, dtype='d')
        return self._minimizable.DoEval(args)

    def _child_fit(self, profile_errors=None):
        assert not profile_errors

        self.setuppars()
        with self.parspecs:
            with FitResult() as fr:
                self._res = minimize(self.call, **self._kwargs)

        fr.set(x=self._res.x, errors=None, fun=self._res.fun,
               success=self._res.success, message=self._res.message,
               minimizer=self.label, nfev=self._res.nit,
               hess_inv = self._res.hess_inv,
               jac = self._res.jac
                )
        self._result = fr.result
        self.patchresult()

        return self.result
