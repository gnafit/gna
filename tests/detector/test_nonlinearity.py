#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from converters import convert
from argparse import ArgumentParser
import constructors as C

parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
parser.add_argument( '-s', '--show', action='store_true' )
opts = parser.parse_args()

def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

def nlfcn( edges ):
    edges = N.asanyarray( edges )
    e1, e2 = 0.5, 2.0
    v1, v2 = 0.4, 1.1
    reg1 = edges<e1
    reg2 = (~reg1)*(edges<e2)
    reg3 = (~reg1)*(~reg2)

    corr = N.zeros_like( edges )
    corr[reg1]=N.NaN
    corr[reg2]=(edges[reg2]-e1)/(e2-e1)*(1.0-v1) + v1
    corr[reg3]=(edges[reg3]-e2)/(12.0-e2)*(v2-1.0) + 1.0

    enew = N.zeros_like( edges )
    enew[reg1]=-2e100
    enew[reg2]=edges[reg2]*corr[reg2]
    enew[reg3]=edges[reg3]*corr[reg3]

    return corr, enew

binwidth=0.05
edges = N.arange( 0.0, 12.0001, binwidth )
corr, edges_m = nlfcn( edges )
edges_m_plot = N.ma.array(edges_m, mask=edges_m<=-1.e100)

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u( 'edep' ) )
ax.set_ylabel( 'Correction' )
ax.set_title( 'Non-linearity correction' )

ax.plot( edges, corr )

savefig( opts.output, suffix='_corr' )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u( 'edep' ) )
ax.set_ylabel( L.u( 'evis' ) )
ax.set_title( 'Non-linearity correction' )

ax.plot( edges, edges_m_plot )
ax.plot( [ edges[0], edges[-1] ], [ edges[0], edges[-1] ], '--' )

savefig( opts.output, suffix='_energy' )

pedges_m = C.Points( edges_m )
ev = [ 1.025, 2.025, 3.025, 5.025, 6.025, 9.025 ]
phist = singularities( ev, edges )
hist = C.Histogram( edges, phist )

histedges = R.HistEdges()
histedges.histedges.hist( hist.hist )

nl = R.HistNonlinearity()
nl.set( histedges.histedges, pedges_m, hist )

smeared = nl.smear.Nvis.data()
print( 'Sum check (diff): {}'.format( phist.sum()-smeared.sum() ) )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('Evis') )
ax.set_ylabel( 'Entries' )
ax.set_title( 'Non-linearity correction' )

lines = plot_hist( edges, smeared )
color = lines[0].get_color()
_, ev_m = nlfcn(ev)
heights = smeared[smeared>0.45]
ax.vlines( ev,   0.0, heights, alpha=0.7, color='red', linestyle='--' )
ax.vlines( ev_m, 0.0, heights, alpha=0.7, color='green', linestyle='--' )

savefig( opts.output, suffix='_evis' )

fig = P.figure()
ax1 = P.subplot( 111 )
ax1.minorticks_on()
ax1.grid()
ax1.set_xlabel( 'Source bins' )
ax1.set_ylabel( 'Target bins' )
ax1.set_title( 'Energy non-linearity matrix' )

mat = convert(nl.getDenseMatrix(), 'matrix')
print( 'Col sum', mat.sum(axis=0) )

mat = N.ma.array( mat, mask= mat==0.0 )
c = ax1.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
add_colorbar( c )

ax1.plot( edges, edges_m_plot, '--', linewidth=0.5, color='white' )
ax1.set_ylim( edges[-1], edges[0] )

savefig( opts.output, suffix='_mat' )

if opts.show:
    P.show()
