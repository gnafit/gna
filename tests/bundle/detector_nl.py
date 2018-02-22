#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import constructors as C
from converters import convert
import numpy as N
from gna.configurator import NestedDict
from gna.bundle import execute_bundle
from physlib import percent

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
opts=parser.parse_args()

#
# Configurate
#
cfg = NestedDict(
        bundle = 'detector_nonlinearity_db_root_v01',
        names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
        filename = 'data/dayabay/tmp/detector_nl_consModel_450itr.root',
        uncertainty = 0.2*percent,
        uncertainty_type = 'relative'
        )

#
# Make input histogram
#
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

#
# Initialize bundle
#
b = execute_bundle( edges=points.single(), cfg=cfg )
pars = [ p for k, p in b.common_namespace.items() if k.startswith('weight') ]
escale = b.common_namespace['escale']

(nonlin,) = b.output_transformations
corr_lsnl = b.storage['lsnl_factor']
corr = b.storage('escale')['factor']

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

for par, name in zip(pars, cfg.names):
    if name!='nominal':
        for par1, name1 in zip(pars[1:], cfg.names[1:]):
            par1.set( name==name1 and 1.0 or 0.0 )

    lines = ax.plot( edges, corr_lsnl.sum.sum.data(), '-', label=name )
    stride = 5
    ax.plot( b.storage('inputs')['edges'][::stride], b.storage('inputs')[name][::stride], 'o', markerfacecolor='none', color=lines[0].get_color() )

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
ax1.set_xlabel( 'Source bins' )
ax1.set_ylabel( 'Target bins' )
ax1.set_title( 'Daya Bay LSNL matrix' )
mat = convert(nonlin.getDenseMatrix(), 'matrix')
print( 'Col sum', mat.sum(axis=0) )

mat = N.ma.array( mat, mask= mat==0.0 )
c = ax1.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
add_colorbar( c )

newe = b.storage('escale')['edges_mod'].product.data()
ax1.plot( edges, newe, '--', color='white', linewidth=0.3 )

savefig( opts.output, suffix='_matrix', dpi=300 )

P.show()