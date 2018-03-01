# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT
import numpy as N
from gna.bindings import patchROOTClass

unctypes = ( ROOT.Variable('double'),  )

namefmt='{name:30}'
valfmt='={val:11.6g}'
centralfmt=' │ {central:11.6g}'
limitsfmt=' ({:g}, {:g})'
centralsigmafmt=' │ {central:11.6g}±{sigma:11.6g}'
relsigmafmt=' [{relsigma:11.6g}%]'
relsigma_len=len(relsigmafmt.format(relsigma=0))

sepstr=' │ '
fixedstr=sepstr+'[fixed]'
freestr=' [free]'
freestr+=' '*(relsigma_len-len(freestr))

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

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    return s

@patchROOTClass( ROOT.Parameter('double'), '__str__' )
def Parameter__str( self, labels=False  ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            )
    limits  = self.limits()

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if self.isFixed():
        s+=fixedstr
        return s

    s+= centralfmt.format(**fmt)

    if limits.size():
        s+=sepstr
        for (a,b) in limits:
            s+=limitsfmt.format(a,b)

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

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if self.isFixed():
        s+=fixedstr
    else:
        s+=centralsigmafmt.format(**fmt)
        if N.isinf(fmt['sigma']):
            s+=freestr
        else:
            if fmt['central']:
                s+=relsigmafmt.format(relsigma=fmt['sigma']/fmt['central']*100.0)
            else:
                s+=' '*relsigma_len

        if limits.size():
            s+=sepstr
            for (a,b) in limits:
                s+=limitsfmt.format(a,b)

    if label:
        s+=sepstr+label

    return s
