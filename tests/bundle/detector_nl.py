#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle.detector_nl import detector_nl_from_file
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist
from gna.env import env
import constructors as C
from converters import convert
import numpy as N

#
# Initialize bundle
#
names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3'  ]
pars = [ env.defparameter( 'weight_'+names[0], central=1.0, sigma=0.0, fixed=True ) ]
for name in names[1:]:
    par = env.defparameter( 'weight_'+name, central=0.0, sigma=1.0 )
    pars.append( par )

escale = env.defparameter( 'weight_escale', central=1.0, sigma=0.02*0.01 )

def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

nbins = 240
edges = N.linspace(0.0, 12.0, nbins+1, dtype='d')
points = C.Points(edges)
phist = singularities( [ 1.225, 2.225, 4.025, 7.025, 9.025 ], edges )
hist = C.Histogram( edges, phist )
ident = R.Identity()
ident.identity.source( points.points )

filename = 'output/detector_nl_consModel_450itr.root'
nonlin, transf = detector_nl_from_file( filename, names, edges=ident.identity.target, debug=True)
corr_lsnl = transf['corr_lsnl']
corr = transf['corr']

#
# Plot curves:
#   - output of the weighted sum (input to the HistNonlinearity)
#   - the curves read from file (as a sanity check)
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( '' )

for par, name in zip(pars, names):
    if name!='nominal':
        for par1, name1 in zip(pars[1:], names[1:]):
            par1.set( name==name1 and 1.0 or 0.0 )

    lines = ax.plot( edges, corr_lsnl.sum.sum.data(), '-', label=name )
    stride = 5
    ax.plot( transf['inputs']['edges'][::stride], transf['inputs'][name][::stride], 'o', markerfacecolor='none', color=lines[0].get_color() )

for par in pars[1:]:
    par.set(0.0)

escale.set(1.1)
ax.plot( edges, corr.sum.sum.data(), '--', label='escale=1.1' )
escale.set(1.0)

ax.legend( loc='lower right' )

#
# Test bundle
#
nonlin.smear.Ntrue( hist.hist )

#
# Plot hists
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'Non-linearity effect' )

smeared = nonlin.smear.Nvis.data().copy()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared.sum() ) )

plot_hist( edges, phist, label='original' )
lines = plot_hist( edges, smeared, label='smeared: nominal' )

ax.legend( loc='upper right' )

#
# Plot matrix
#
fig = P.figure()
ax1 = P.subplot( 111 )
ax1.minorticks_on()
ax1.grid()
ax1.set_xlabel( '' )
ax1.set_ylabel( '' )
ax1.set_title( '' )
mat = convert(nonlin.getDenseMatrix(), 'matrix')
print( 'Col sum', mat.sum(axis=0) )

mat = N.ma.array( mat, mask= mat==0.0 )
c = ax1.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
add_colorbar( c )

newe = transf['newe'].product.data()
ax1.plot( edges, newe, '--', color='white' )

P.show()
