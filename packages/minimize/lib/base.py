from __future__ import print_function
import numpy as np
import time
from collections import OrderedDict

class FitResult(object):
    def __init__(self):
        self._result = OrderedDict()

    @property
    def result(self):
        return self._result

    def __enter__(self):
        self._wall  = time.time()
        self._clock = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clock = time.clock() - self._clock
        self._wall  = time.time()  - self._wall

    def set(self, x, errors, fun, success, message, minimizer, nfev, **kwargs):
        result = self._result

        result['x']         = x
        result['errors']    = errors
        result['fun']       = fun
        result['success']   = success
        result['message']   = message
        result['nfev']      = nfev
        result['minimizer'] = minimizer
        hess_inv = result['hess_inv']  = kwargs.pop('hess_inv', None)
        result['jac']       = kwargs.pop('jac', None)

        if errors is None and hess_inv is not None:
            result['errors'] = np.diag(hess_inv)*2.0

        result['clock'] = self._clock
        result['wall']  = self._wall


class MinimizerBase(object):
    _minimizable = None
    _parspecs    = None
    _result      = None
    def __init__(self, statistic, minpars):
        self.statistic = statistic
        self.parspecs = minpars

    @property
    def statistic(self):
        return self._statistic

    @statistic.setter
    def statistic(self, statistic):
        self._statistic = statistic
        self._minimizable = None

    @property
    def parspecs(self):
        return self._parspecs

    @parspecs.setter
    def parspecs(self, parspecs):
        self._parspecs = parspecs

    @property
    def result(self):
        return self._result

    def fit(self, profile_errors=[]):
        raise Exception('Calling unimplemented base.fit() method')

    def patchresult(self):
        names = list(self._parspecs.names())
        result = self._result
        result['xdict']      = OrderedDict(zip(names, (float(x) for x in self.result['x'])))
        result['errorsdict'] = OrderedDict(zip(names, (float(e) for e in self.result['errors'])))
        result['names']      = names
        result['npars']      = self._parspecs.nvariable()
        result['nfree']      = self._parspecs.nfree()
        result['nfixed']     = self._parspecs.nfixed()
        result['nconstrained'] = self._parspecs.nconstrained()

    def evalstatistic(self):
        with FitResult() as fr:
            fun = self._statistic()

        fr.set(x=[], errors=[], fun=fun,
               success=True, message='stastitics evaluation (no parameters)',
               minimizer='none', nfev=1)
                )
        self._result = fr.result
        self.patchresult()

        return self.result
