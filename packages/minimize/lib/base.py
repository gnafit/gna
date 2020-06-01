from __future__ import print_function
import numpy as np
import time
from collections import OrderedDict

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
