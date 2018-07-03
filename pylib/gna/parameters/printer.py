# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT
import numpy as N
from gna.bindings import patchROOTClass

unctypes = ( ROOT.Variable('double'),  )

namefmt='{name:30}'
valfmt='={val:11.6g}'
centralfmt='{central:11.6g}'
limitsfmt=' ({:g}, {:g})'
centralsigmafmt='{central:11.6g}±{sigma:11.6g}'
relsigmafmt=' [{relsigma:11.6g}%]'

centralsigma_len=len(centralsigmafmt.format(central=0, sigma=0))
relsigma_len=len(relsigmafmt.format(relsigma=0))

centralrel_empty =(centralsigma_len+relsigma_len-1)*' '
sepstr=' │ '
fixedstr='[fixed]'
fixedstr+=' '*(centralsigma_len+relsigma_len-1-len(fixedstr))
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
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if labels:
        s+=sepstr+centralrel_empty+sepstr
    if label:
        s+=label

    return s

@patchROOTClass( ROOT.Parameter('double'), '__str__' )
def Parameter__str( self, labels=False  ):
    fmt = dict(
            name    = self.name(),
            val     = self.value(),
            central = self.central(),
            )
    limits  = self.limits()
    if label:
        s+=sepstr+label

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if self.isFixed():
        s+=sepstr+fixedstr
    else:
        s+= sepstr+centralfmt.format(**fmt)

        if limits.size():
            s+=sepstr
            for (a,b) in limits:
                s+=limitsfmt.format(a,b)

    if labels:
        s+=sepstr
    if label:
        s+=label

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
        s+=sepstr+fixedstr
    else:
        s+=sepstr + centralsigmafmt.format(**fmt)
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

    if labels:
        s+=sepstr
    if label:
        s+=label

    return s
