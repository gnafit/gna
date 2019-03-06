import ROOT
from argparse import Namespace
import numpy as np
import time
import spec
from collections import OrderedDict

class Minuit(ROOT.TMinuitMinimizer):
    def __init__(self, statistic, pars=[]):
        super(Minuit, self).__init__()

        ROOT.TMinuitMinimizer.UseStaticMinuit(False)

        self.statistic = statistic

        self.pars = []
        self.spec = {}
        self.addpars(pars)

        self.result = None

    @property
    def statistic(self):
        return self._statistic

    @statistic.setter
    def statistic(self, statistic):
        self._statistic = statistic
        self._reset = True

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, newspec):
        self._spec = spec.merge({}, newspec)
        self._reset = True

    @property
    def tolerance(self):
        return self.Tolerance()

    @tolerance.setter
    def tolerance(self, tolerance):
        self.SetTolerance(tolerance)

    def addpars(self, pars):
        self.pars.extend(pars)
        self._reset = True

    def setuppar(self, i, par, parspec):
        value = parspec.get('value', par.value())
        if isinstance(value, spec.dynamicvalue):
            value = value.value(par)
        step = parspec.get('step', par.step())
        if step==0:
            raise Exception( '"%s" initial step is undefined. Specify its sigma explicitly.'%par.name() )
        vmin, vmax = parspec.get('limits', [float('-inf'), float('+inf')])
        fixed = parspec.get('fixed', False)

        if fixed:
            self.SetFixedVariable(i, par.name(), value)
        elif (vmin, vmax) == (float('-inf'), float('+inf')):
            self.SetVariable(i, par.name(), value, step)
        elif vmax == float('+inf'):
            self.SetLowerLimitedVariable(i, par.name(), value, step, vmin)
        elif vmin == float('-inf'):
            self.SetUpperLimitedVariable(i, par.name(), value, step, vmax)
        else:
            self.SetLimitedVariable(i, par.name(), value, step, vmin, vmax)

    def setuppars(self):
        self._minimizable = ROOT.Minimizable(self.statistic)
        for par in self.pars:
            self._minimizable.addParameter(par)
        self.SetFunction(self._minimizable)

        spec = self.spec
        for i, par in enumerate(self.pars):
            self.setuppar(i, par, spec.get(par, {}))

        self._reset = False

    def affects(self, par):
        if par not in self.pars:
            return False
        spec = self.spec.get(par, {})
        fixed = spec.get('fixed', False)
        if not fixed:
            return True
        if 'value' in spec:
            return True
        return False

    def freepars(self):
        res = []
        for par in self.pars:
            spec = self.spec.get(par, {})
            if spec.get('fixed', False):
                continue
            res.append(par)
        return res

    def evalstatistic(self):
        wall = time.time()
        clock = time.clock()
        value = self._statistic()
        clock = time.clock() - clock
        wall = time.time() - wall

        x = [par.value() for par in self.pars]
        resultdict = {
            'x': np.array(x),
            'errors': np.zeros_like(x),
            'success': True,
            'fun': value,
            'nfev': 1,
            'maxcv': 0.0,
            'wall': wall,
            'cpu': clock,
        }
        self.result = Namespace(**resultdict)
        self._patchresult()
        return self.result

    def fit(self):
        if not self.pars:
            return self.evalstatistic()

        if self._reset:
            self.setuppars()
        else:
            self.SetFunction(self._minimizable)

        wall = time.time()
        clock = time.clock()
        self.Minimize()
        clock = time.clock() - clock
        wall = time.time() - wall

        argmin = np.frombuffer(self.X(), dtype=float, count=self.NDim())
        errors = np.frombuffer(self.Errors(), dtype=float, count=self.NDim())

        resultdict = {
            'x': argmin.copy(),
            'errors': errors.copy(),
            'success': not self.Status(),
            'fun': self.MinValue(),
            'nfev': self.NCalls(),
            'maxcv': self.Tolerance(),
            'wall': wall,
            'cpu': clock,
        }
        self.result = Namespace(**resultdict)
        self._patchresult()
        return self.result

    def _patchresult(self):
        names = [self.VariableName(i) for i in range(self.NDim())]
        self.result.xdict      = OrderedDict(zip(names, (float(x) for x in self.result.x)))
        self.result.errorsdict = OrderedDict(zip(names, (float(e) for e in self.result.errors)))
        self.result.names = names
        self.result.npars = int(self.NDim())
        self.result.nfev = int(self.result.nfev)
        self.result.npars = int(self.result.npars)

    def __call__(self):
        res = self.fit()
        if not res.success:
            return None
        return res.fun
