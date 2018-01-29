# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT
from gna.bindings import patchROOTClass

unctypes = ( ROOT.Variable('double'),  )

def print_parameters( ns, recursive=True ):
    header = False
    for name, var in ns.iteritems():
        if isinstance( ns.storage[name], basestring ):
            print(u'  {name:30}-> {target}'.format( name=name, target=ns.storage[name] ))
            continue
        if not isinstance( var, unctypes ):
            continue
        if not header:
            print("Variables in namespace '%s'"%ns.path)
            header=True
        print(end='  ')
        print(var)
    if recursive:
        for sns in ns.namespaces.itervalues():
            print_parameters( sns )

@patchROOTClass( ROOT.Variable('double'), '__str__' )
def Variable__str( self ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            )

    s= '{name:30}'.format(**fmt)
    s+='={val:10.6g}'.format(**fmt)

    return s

@patchROOTClass( ROOT.Parameter('double'), '__str__' )
def Parameter__str( self ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            )
    limits  = self.limits()

    s= '{name:30}'.format(**fmt)
    s+='={val:10.6g}'.format(**fmt)

    if self.isFixed():
        s+=' │ [fixed]'
        return s

    s+= ' │ {central:10.6g}'.format(**fmt)

    if limits.size():
        s+=' │'
        for (a,b) in limits:
            s+=' (%g, %g)'%(a,b)

    return s

@patchROOTClass( ROOT.GaussianParameter('double'), '__str__' )
def Parameter__str( self ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            sigma   = self.sigma(),
            )
    limits  = self.limits()

    s= '{name:30}'.format(**fmt)
    s+='={val:10.6g}'.format(**fmt)

    if self.isFixed():
        s+=' │ [fixed]'
        return s

    s+=' │ {central:10.6g}±{sigma:10.6g}'.format(**fmt)
    if fmt['central']:
        s+=' [{relsigma:10.6g}%]'.format(relsigma=fmt['sigma']/fmt['central']*100.0)
    else:
        s+=' '*14

    if limits.size():
        s+=' │'
        for (a,b) in limits:
            s+=' (%g, %g)'%(a,b)

    return s
