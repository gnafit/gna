# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import ROOT
from argparse import Namespace
import numpy as np
import time
from . import spec
from collections import OrderedDict

class Minuit(ROOT.TMinuitMinimizer):
    def __init__(self, statistic, pars=[]):
        super(Minuit, self).__init__()

        ROOT.TMinuitMinimizer.UseStaticMinuit(False)

        self.statistic = statistic

        self.pars = []
        self.parsdict = OrderedDict()
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
        for par in pars:
            self.pars.append(par)
            self.parsdict[par.qualifiedName()] = par
        self._reset = True

    def setuppar(self, i, par, parspec):
        value = parspec.get('value', par.value())
        if isinstance(value, spec.dynamicvalue):
            value = value.value(par)
        step = parspec.get('step', par.step())
        qualifiedName = par.qualifiedName()
        if step==0:
            raise Exception( '"%s" initial step is undefined. Specify its sigma explicitly.'%qualifiedName )

        try:
            vmin, vmax = parspec['limits']
        except KeyError:
            try:
                vmin, vmax = tuple(par.limits()[0])
            except IndexError:
                vmin, vmax = float('-inf'), float('+inf')

        fixed = parspec.get('fixed', False)

        if fixed:
            self.SetFixedVariable(i, qualifiedName, value)
        elif (vmin, vmax) == (float('-inf'), float('+inf')):
            self.SetVariable(i, qualifiedName, value, step)
        elif vmax == float('+inf'):
            self.SetLowerLimitedVariable(i, qualifiedName, value, step, vmin)
        elif vmin == float('-inf'):
            self.SetUpperLimitedVariable(i, qualifiedName, value, step, vmax)
        else:
            self.SetLimitedVariable(i, qualifiedName, value, step, vmin, vmax)

    def setuppars(self):
        self._minimizable = ROOT.Minimizable(self.statistic)
        for par in self.pars:
            self._minimizable.addParameter(par)
        self.SetFunction(self._minimizable)

        spec = self.spec
        for i, par in enumerate(self.pars):
            self.setuppar(i, par, spec.get(par, {}))

        self._reset = False

    def resetpars(self):
        if self._reset:
            return
        spec = self.spec
        for i, par in enumerate(self.pars):
            self.setuppar(i, par, spec.get(par, {}))

    def fixpar(self, name, val=None, fixed=True):
        try:
            par = self.parsdict[name]
        except:
            raise Exception('Variable {} is not known to minimizer'.format(name))

        parspec=self.spec.setdefault(par, {})
        parspec['fixed']=fixed
        if val is not None:
            parspec['value']=val

        self._reset = True

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

    def pushpars(self):
        for par in self.pars:
            par.push()

    def poppars(self):
        for par in self.pars:
            par.pop()

    def fit(self, profile_errors=[]):
        if not self.pars:
            return self.evalstatistic()

        if self._reset:
            self.setuppars()
        else:
            self.SetFunction(self._minimizable)

        self.pushpars()

        wall = time.time()
        clock = time.clock()
        self.Minimize()
        clock = time.clock() - clock
        wall = time.time() - wall

        self.poppars()

        argmin = np.frombuffer(self.X(), dtype=float, count=self.NDim())
        errors = np.frombuffer(self.Errors(), dtype=float, count=self.NDim())

        resultdict = {
            'x': argmin.tolist(),
            'errors': errors.tolist(),
            'success': not self.Status(),
            'fun': self.MinValue(),
            'nfev': self.NCalls(),
            'maxcv': self.Tolerance(),
            'wall': wall,
            'cpu': clock,
        }
        self.result = Namespace(**resultdict)
        self._patchresult()

        if profile_errors:
            self.profile_errors(profile_errors, self.result)

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

    def profile_errors(self, names, fitresult):
        errs = fitresult.errors_profile = OrderedDict()
        if names:
            print('Caclulating statistics profile for:', end=' ')
        for name in names:
            if isinstance(name, int):
                idx = name
                name = self.VariableName(idx)
            else:
                idx = self.result.names.index(name)
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
