import numpy as np
import time
import ROOT

class FitResult:
    def __init__(self):
        self._result = dict()

    @property
    def result(self):
        return self._result

    def __enter__(self):
        self._wall  = time.time()
        self._clock = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clock = time.perf_counter() - self._clock
        self._wall  = time.time()  - self._wall

    def set(self, x, errors, fun, success, message, minimizer, nfev, **kwargs):
        result = self._result

        result['fun']       = fun
        result['success']   = success
        result['message']   = message
        result['nfev']      = nfev
        result['minimizer'] = minimizer
        result['clock'] = self._clock
        result['wall']  = self._wall
        result['x']         = x
        result['errors']    = errors
        hess_inv = result['hess_inv']  = kwargs.pop('hess_inv', None)
        result['jac']       = kwargs.pop('jac', None)

        if errors is None and hess_inv is not None:
            result['errors'] = np.diag(hess_inv)*2.0

        result.update(kwargs)

class MinimizerBase(object):
    _name: str
    _label: str  = ''
    _minimizable = None
    _parspecs    = None
    _result      = None
    def __init__(self, statistic, minpars, name: str='', *, minimizable_verbose=False):
        self.statistic = statistic
        self.parspecs = minpars
        self._name = name

        if minimizable_verbose:
            self._minimizable_class = ROOT.MinimizableVerbose
        else:
            self._minimizable_class = ROOT.Minimizable

    @property
    def statistic(self):
        return self._statistic

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    @statistic.setter
    def statistic(self, statistic):
        self._statistic = statistic
        self._minimizable = None

    def update_minimizable(self):
        if self._minimizable is None or self.parspecs.resized:
            self._minimizable = self._minimizable_class(self.statistic)

            for parspec in self.parspecs.specs():
                self._minimizable.addParameter(parspec.par)

        return self._minimizable

    @property
    def parspecs(self):
        return self._parspecs

    @parspecs.setter
    def parspecs(self, parspecs):
        self._parspecs = parspecs

    @property
    def result(self):
        return self._result

    def _child_fit(self, **kwargs):
        raise Exception('Calling unimplemented base.fit() method')

    def fit(self, **kwargs):
        if len(self.parspecs)==0:
            return self.evalstatistic()

        return self._child_fit(**kwargs)

    def patchresult(self):
        names = list(self._parspecs.names())
        result = self._result
        result['npars']         = self._parspecs.nvariable()
        result['nfree']         = self._parspecs.nfree()
        result['nconstrained']  = self._parspecs.nconstrained()
        fixed = result['fixed'] = self._parspecs.fixed()
        result['nfixed']        = len(fixed)
        result['x'] = result.pop('x')
        result['errors'] = result.pop('errors')
        result['names']         = names
        result['xdict']      = dict(zip(names, (float(x) for x in self.result['x'])))
        if self.result['errors'] is not None:
            result['errorsdict'] = dict(zip(names, (float(e) for e in self.result['errors'])))
        else:
            result['errorsdict'] = {}

    def evalstatistic(self):
        with FitResult() as fr:
            fun = self._statistic()

        fr.set(x=[], errors=[], fun=fun,
               success=True, message='stastitics evaluation (no parameters)',
               minimizer='none', nfev=1
               )
        self._result = fr.result
        self.patchresult()

        return self.result

    def saveresult(self, loc):
        pass
