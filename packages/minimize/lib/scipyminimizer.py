#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from scipy.optimize import minimize

class SciPyMinimizer(object):
    _minimizable = None

    # SciPy specific options
    method = 'BFGS'
    err    =  1.0
    fcn    = None
    varnames = []
    x0       = []
    res      = None
    errors   = []
    limits   = []
    def __init__(self, statistic, minpars, method=None):
        if method:
            self.method = method

        ROOT.TMinuitMinimizer.UseStaticMinuit(False)
        self.statistic = statistic
        self.parspecs = minpars
        self.result = None

            # self.x0.append( val )
            # self.varnames.append( name )
            # self.errors.append( 0.0 )
            # self.limits.append( (None, None) )

    def Minimize(self):
        assert self.fcn and self.x0
        clock = time.clock()
        if self.method=='Anneal':
            from operator import itemgetter
            lower = [ itemgetter(0)(i) for i in self.limits ]
            upper = [ itemgetter(1)(i) for i in self.limits ]
            myopts = { 'maxiter'      : 500,
                       'ftol'         : 1e-6,
                       'lower'        : lower,
                       'upper'        : upper,
                       'disp'         : True
                      }
            self.res = minimize( self.call, self.x0, method=self.method, options=myopts )
        else:
            self.res = minimize( self.call, self.x0, method=self.method, bounds=self.limits )
        ##end if
        clock = time.clock() - clock
        self.res.time = clock
        return self.res
    ##end def Minimize

    def PrintResults(self, fcn=None):
        print( 'Print minimization results (%s):'%self.method )
        for i, (name, val) in enumerate( zip( self.varnames, self.res.x ) ):
            print( i, name, val )
        ##end for i, (name, val)

        print( 'Chi2 at minimum =  %6.2f'%self.MinValue() )
        if fcn:
            print( 'ndf = %i (ndata=%i, npunish=%i, npars=%i)'%( fcn.NdfResult(), fcn.NdfData()
                                                               , fcn.NdfPunishment(), fcn.NDim() ) )
            print( 'Chi2/ndf =  %6.2f'%( self.MinValue()/fcn.NdfResult() ) )
        print( 'Ncalls', self.NCalls() )
    ##end def PrintResults

    @property
    def statistic(self):
        return self._statistic

    @statistic.setter
    def statistic(self, statistic):
        self._statistic = statistic
        self._minimizable = None

    @property
    def tolerance(self):
        return self.Tolerance()

    @tolerance.setter
    def tolerance(self, tolerance):
        self.SetTolerance(tolerance)

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

        self.parspecs.poppars()

        argmin = np.frombuffer(self.X(), dtype=float, count=self.NDim())
        errors = np.frombuffer(self.Errors(), dtype=float, count=self.NDim())

        self.result = {
            'x': argmin.tolist(),
            'errors': errors.tolist(),
            'success': not self.Status(),
            'fun': self.MinValue(),
            'nfev': self.NCalls(),
            'maxcv': self.Tolerance(),
            'wall': wall,
            'cpu': clock,
        }
        self._patchresult()

        if profile_errors:
            self.profile_errors(profile_errors, self.result)

        return self.result

    def _patchresult(self):
        names = [self.VariableName(i) for i in range(self.NDim())]
        self.result['xdict']      = OrderedDict(zip(names, (float(x) for x in self.result['x'])))
        self.result['errorsdict'] = OrderedDict(zip(names, (float(e) for e in self.result['errors'])))
        self.result['names'] = names
        self.result['npars'] = int(self.NDim())
        self.result['nfev'] = int(self.result['nfev'])
        self.result['npars'] = int(self.result['npars'])

