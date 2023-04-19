#!/usr/bin/env python

from scipy.optimize import minimize
from minimize.lib.base import MinimizerBase, FitResult
import numpy as np
from typing import Optional

class SciPyMinimizer(MinimizerBase):
    _label: str       = 'scipy'
    _method: str      = 'BFGS'
    _kwargs: dict     = {}
    _scipy_opts: dict = {}
    def __init__(self, statistic, minpars, *, method: Optional[str]=None, scipy_opts: dict={}, **kwargs):
        MinimizerBase.__init__(self, statistic, minpars, **kwargs)
        if method:
            self._method = method
        self._scipy_opts = scipy_opts

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

        self._kwargs.update(self._scipy_opts)

    def call(self, args0):
        """Evaluate the function for a given set of values"""
        args = np.ascontiguousarray(args0, dtype='d')
        return self._minimizable.DoEval(args)

    def _child_fit(self, **kwargs):
        assert not kwargs

        self.setuppars()
        with self.parspecs:
            with FitResult() as fr:
                self._res = minimize(self.call, **self._kwargs)

        try:
            hess_inv = self._res.hess_inv
        except AttributeError:
            hess_inv = None

        try:
            jac = self._res.jac
        except AttributeError:
            jac = None
        fr.set(x=self._res.x, errors=None, fun=float(self._res.fun),
               success=self._res.success, message=self._res.message,
               minimizer=self.label, nfev=self._res.nit,
               hess_inv = hess_inv,
               jac = jac
                )
        self._result = fr.result
        self.patchresult()

        return self.result
