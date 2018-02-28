# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT
import numpy as N
from gna.bindings import patchROOTClass

unctypes = ( ROOT.Variable('double'),  )

def print_parameters( ns, recursive=True, labels=False ):
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
        print(var.__str__(labels=labels))
    if recursive:
        for sns in ns.namespaces.itervalues():
            print_parameters( sns, recursive=recursive, labels=labels )

@patchROOTClass( ROOT.Variable('double'), '__str__' )
def Variable__str( self, labels=False ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            )

    s= '{name:30}'.format(**fmt)
    s+='={val:10.6g}'.format(**fmt)

    return s

@patchROOTClass( ROOT.Parameter('double'), '__str__' )
def Parameter__str( self, labels=False  ):
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
def Parameter__str( self, labels=False  ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            sigma   = self.sigma()
            )
    limits  = self.limits()
    label = self.label()
    if not labels or label=='value':
        label=''

    s= '{name:30}'.format(**fmt)
    s+='={val:10.6g}'.format(**fmt)

    if self.isFixed():
        s+=' │ [fixed]'
    else:
        s+=' │ {central:10.6g}±{sigma:10.6g}'.format(**fmt)
        if N.isinf(fmt['sigma']):
            s+=' [free]'+' '*7
        else:
            if fmt['central']:
                s+=' [{relsigma:10.6g}%]'.format(relsigma=fmt['sigma']/fmt['central']*100.0)
            else:
                s+=' '*14

        if limits.size():
            s+=' │'
            for (a,b) in limits:
                s+=' (%g, %g)'%(a,b)

    if label:
        s+=' │ '+label

    return s
