from __future__ import print_function
import ROOT
import numpy as np
import time
from packages.minimize.lib.base import MinimizerBase
from collections import OrderedDict

class MinuitBase(MinimizerBase):
    _label = 'TMinuit?'
    def __init__(self, statistic, minpars):
        MinimizerBase.__init__(self, statistic, minpars)

    @property
    def tolerance(self):
        return self.Tolerance()

    @tolerance.setter
    def tolerance(self, tolerance):
        self.SetTolerance(tolerance)

    @property
    def label(self):
        return self._label

    def setuppars(self):
        if not self.parspecs.modified:
            return

        if self._minimizable is None or self.parspecs.resized:
            self._minimizable = ROOT.Minimizable(self.statistic)

            for parspec in self.parspecs.specs():
                self._minimizable.addParameter(parspec.par)

        self.SetFunction(self._minimizable)

        for i, (name, parspec) in enumerate(self.parspecs.items()):
            self.setuppar(i, name, parspec)

        self.parspecs.resetstatus()

    def setuppar(self, i, name, parspec):
        vmin, vmax = parspec.vmin, parspec.vmax
        if parspec.fixed:
            self.SetFixedVariable(i, name, parspec.value)
        elif (vmin, vmax) == (None, None):
            self.SetVariable(i, name, parspec.value, parspec.step)
        elif vmax is None:
            self.SetLowerLimitedVariable(i, name, parspec.value, parspec.step, vmin)
        elif vmin is None:
            self.SetUpperLimitedVariable(i, name, parspec.value, parspec.step, vmax)
        else:
            self.SetLimitedVariable(i, name, parspec.value, parspec.step, vmin, vmax)

    # def resetpars(self):
        # if self._reset:
            # return
        # spec = self.spec
        # for i, par in enumerate(self.parspecs):
            # self.setuppar(i, par, spec.get(par, {}))

    # def evalstatistic(self):
        # wall = time.time()
        # clock = time.clock()
        # value = self._statistic()
        # clock = time.clock() - clock
        # wall = time.time() - wall

        # x = [par.value() for par in self.parspecs]
        # resultdict = {
            # 'x': np.array(x),
            # 'errors': np.zeros_like(x),
            # 'success': True,
            # 'fun': value,
            # 'nfev': 1,
            # 'maxcv': 0.0,
            # 'wall': wall,
            # 'cpu': clock,
        # }
        # self.result = Namespace(**resultdict)
        # self._patchresult()
        # return self.result


    def fit(self, profile_errors=[]):
        assert self.parspecs
        # if not self.parspecs:
            # return self.evalstatistic()

        self.setuppars()

        self.parspecs.pushpars()

        wall = time.time()
        clock = time.clock()
        self.Minimize()
        clock = time.clock() - clock
        wall = time.time() - wall

        argmin = np.frombuffer(self.X(), dtype=float, count=self.NDim())
        errors = np.frombuffer(self.Errors(), dtype=float, count=self.NDim())

        self._result = {
            'x':         argmin,
            'errors':    errors,
            'success':   not self.Status(),
            'message':   '',
            'fun':       self.MinValue(),
            'nfev':      int(self.NCalls()),
            # 'maxcv':     self.Tolerance(),
            'wall':      wall,
            'cpu':       clock,
            'minimizer': self.label,
            'hess_inv':  None,
            'jac':       None,
        }
        self.patchresult()

        if profile_errors:
            self.profile_errors(profile_errors, self.result)

        self.parspecs.poppars()

        return self.result

    # def __call__(self):
        # res = self.fit()
        # if not res.success:
            # return None
        # return res.fun

    def profile_errors(self, names, fitresult):
        errs = fitresult['errors_profile'] = OrderedDict()
        if names:
            print('Caclulating statistics profile for:', end=' ')
        for name in names:
            if isinstance(name, int):
                idx = name
                name = self.VariableName(idx)
            else:
                idx = self.result['names'].index(name)
            print(name, end=', ')
            left, right = self.get_profile_error(idx=idx)
            errs[name] = [left.tolist(), right.tolist()]

    def get_profile_error(self, name=None, idx=None, verbose=False):
        if idx==None:
            idx = self.VariableIndex( name )

        if not name:
            name = self.VariableName( idx )

        if verbose:
            print( '    variable %i %s'%( idx, name ), end='' )

        low, up = np.zeros( 1, dtype='d' ), np.zeros( 1, dtype='d' )
        try:
            self.GetMinosError( idx, low, up )
        except:
            print( 'Minuit error!' )
            return [ 0.0, 0.0 ]

        print( ':', low[0], up[0] )
        return [ low[0], up[0] ]
