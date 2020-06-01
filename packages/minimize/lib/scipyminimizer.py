#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import ROOT

class SciPyMinimizer(object):
    """Mimic ROOT::Math::Minimizer functions"""
    method = 'BFGS'
    err    =  1.0
    fcn    = None
    varnames = []
    x0       = []
    res      = None
    errors   = []
    limits = []
    def __init__(self, method=None):
        if method: self.method = method
    ##end def __init__

    def SetErrorDef( self, err ):
        self.err = err
    ##end if

    def SetFunction( self, fcn ):
        self.fcn = fcn
    ##end if

    def SetVariable(self, i, name, val, err):
        if len( self.x0 )>i:
            self.x0[i] = val
            self.varnames[i] = name
            self.limits[i] = (None, None)
        else:
            self.x0.append( val )
            self.varnames.append( name )
            self.errors.append( 0.0 )
            self.limits.append( (None, None) )
        ##end if
    ##end def SetVariable

    def SetLimitedVariable(self, i, name, val, err, vmin, vmax):
        if len( self.x0 )>i:
            self.x0[i] = val
            self.varnames[i] = name
            self.limits[i] = (float(vmin), float(vmax))
        else:
            self.x0.append( val )
            self.varnames.append( name )
            self.errors.append( 0.0 )
            self.limits.append( (float(vmin), float(vmax)) )
        ##end if
    ##end def SetVariable

    def setPars(self, pars, *args, **kwargs):
        minimizerSetPars( self, pars, *args, **kwargs )
    ##end def setPars

    def readPars(self, pars, *args, **kwargs):
        minimizerReadPars( self, pars, *args, **kwargs )
    ##end def readPars

    def call(self, args):
        self.fcn.GetPars().SetValue( args )
        self.fcn.GetPars().Update()
        res = self.fcn.Eval()
        # print( args, res )
        return res
    ##end def call

    def Minimize(self):
        assert self.fcn and self.x0
        from scipy.optimize import minimize
        clock = time.clock()
        if self.method=='Anneal':
            from operator import itemgetter
            lower = [ itemgetter(0)(i) for i in self.limits ]
            upper = [ itemgetter(1)(i) for i in self.limits ]
            myopts = { 'schedule'     : 'boltzmann'
                      ,'maxfev'       : None
                      ,'maxiter'      : 500
                      ,'maxaccept'    : None
                      ,'ftol'         : 1e-6
                      ,'T0'           : None
                      ,'Tf'           : 1e-12
                      ,'boltzmann'    : 1.0
                      ,'learn_rate'   : 0.5
                      ,'quench'       : 1.0
                      ,'m'            : 1.0
                      ,'n'            : 1.0
                      ,'lower'        : lower
                      ,'upper'        : upper
                      ,'dwell'        : 250
                      ,'disp'         : True
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

    def printResult(self, fun):
        return self.PrintResults( fun )
    ##end def printResult

    def MinValue(self):
        return self.res.fun
    ##end def MinValue

    def X(self):
        return self.res.x
    ##end def X

    def Errors(self):
        return self.errors
    ##end def function

    def Status(self):
        return self.res.status
    ##end def Status

    def NCalls(self):
        return self.res.nfev
    ##end def Status
##end class SciPyMinimizer


