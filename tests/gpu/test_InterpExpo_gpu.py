#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.constructors import Points
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig
from gna import context, bindings

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-m', '--mode', default='expo', choices=['expo', 'linear'], help='Interpolation mode' )
parser.add_argument( '-U', '--underflow', default="", choices=['constant', 'extrapolate'] )
parser.add_argument( '-O', '--overflow', default="", choices=['constant', 'extrapolate'] )
parser.add_argument( '-o', '--output' )
opts = parser.parse_args()

segments   = N.arange(1.0, 10.1, 1.5, dtype='d')

points   = N.stack([N.linspace(0.0+i, 12.+i, 61, dtype='d') for i in [0, -0.1, 0.1, 0.3, 0.5]]).T

ndata=70
with context.manager(ndata) as manager:
    segments_t = Points(segments)
    points_t = Points(points)
    fcn = N.exp( -(segments-segments[0])*0.5 )
    fcn = N.exp(segments**(-0.5))
    fcn_t = Points(fcn)
    
    print( 'Edges', segments )
    print( 'Points', points )
    print( 'Fcn', fcn )
    
    if opts.mode=='expo':
         ie = R.InterpExpo()
    else:
        ie = R.InterpLinear()
    
    ie.interpolate(segments_t, fcn_t, points_t)
#    ie.interp.switchFunction("gpu")
    ie.printtransformations()
    seg_idx = ie.insegment.insegment.data()
    print( 'Segments', seg_idx )
    
    #ns.materializeexpressions()
    #pars = tuple(par.getVariable() for (name,par) in ns.walknames())
    #manager.setVariables(C.stdvector(pars))
    
res = ie.interp.interp.data()
print( 'Result', res )
#ie.print()
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_title( 'Expo' )

ax.plot( segments, fcn, 'o-', markerfacecolor='none', label='coarse function', linewidth=0.5, alpha=0.5 )

markers='os*^vh'
for i, (p, r) in enumerate(zip(points.T, res.T)):
    ax.plot( p, r, '.', label='interpolation, col %i'%i, marker=markers[i], markersize=1.5 )
    break

ax.legend(loc='upper right')
# ax.set_yscale('log')

savefig(opts.output)

P.show()
