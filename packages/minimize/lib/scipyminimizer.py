#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from scipy.optimize import minimize
import ROOT
import numpy as np

class SciPyMinimizer(object):
    _minimizable = None
    _method = 'BFGS'
    _kwargs = None
    def __init__(self, statistic, minpars, method=None):
        if method:
            self._method = method

        self.statistic = statistic
        self.parspecs = minpars
        self.result = None

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def label(self):
        return '.'.join(('scipy', self._method))

    def setuppars(self):
        if self.kwargs is not None and not self.parspecs.modified:
            return

        if self._minimizable is None or self.parspecs.resized:
            self._minimizable = ROOT.Minimizable(self.statistic)

            for parspec in self.parspecs.specs():
                self._minimizable.addParameter(parspec.par)

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

            if bounds!=(None, None):
                bounded=True

        if not bounded:
            self.bounds=None

        self.parspecs.resetstatus()

    def call(self, args0):
        """Evaluate the function for a given set of values"""
        args = np.ascontiguousarray(args0, dtype='d')
        return self._minimizable.DoEval(args)

    def fit(self, profile_errors=None):
        assert self.parspecs
        assert not profile_errors

        self.parspecs.pushpars()
        self.setuppars()

        wall = time.time()
        clock = time.clock()
        self._res = minimize(self.call, **self._kwargs)
        clock = time.clock() - clock
        wall = time.time() - wall

        self.parspecs.poppars()

        self._result = {
            'x':       self._res.x,
            'errors':  None,
            'success': self._res.success,
            'message': self._res.message,
            'fun':     self._res.fun,
            'nfev':    self._res.nit,
            'maxcv':   self._res.maxcv,
            'wall':    wall,
            'cpu':     clock,
            'label':   self.label
        }
        self._patchresult()

        return self.result


