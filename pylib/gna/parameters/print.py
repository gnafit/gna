# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass

@patchROOTClass( ROOT.Uncertain('double'), 'print' )
def Uncertain__print( self ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            sigma   = self.sigma(),
            )

    print( '{name:30}'.format(**fmt), end='' )
    print( '={val:10.6g}'.format(**fmt), end='' )

    print( ' | {central:10.6g}±{sigma:10.6g}'.format(**fmt), end='' )
    if fmt['central']:
        print( ' [{relsigma:10.6g}%]'.format(relsigma=fmt['sigma']/fmt['central']), end='' )

    print()

@patchROOTClass( ROOT.Parameter('double'), 'print' )
def Parameter__print( self ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            sigma   = self.sigma(),
            )
    limits  = self.limits()

    print( '{name:30}'.format(**fmt), end='' )
    print( '={val:10.6g}'.format(**fmt), end='' )

    if self.isFixed():
        print( ' [fixed]' )
        return

    print( ' | {central:10.6g}±{sigma:10.6g}'.format(**fmt), end='' )
    if fmt['central']:
        print( ' [{relsigma:10.6g}%]'.format(relsigma=fmt['sigma']/fmt['central']), end='' )

    if limits.size():
        print( ' |', end='' )
        for (a,b) in limits:
            print( ' (%g, %g)'%(a,b), end='' )

    print()
