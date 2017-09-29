#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from converters import convert
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
parser.add_argument( '-s', '--show', action='store_true' )
opts = parser.parse_args()

def axes( title, ylabel='' ):
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( L.u('evis') )
    ax.set_ylabel( ylabel )
    ax.set_title( title )
    return ax

def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

binwidth=0.05
edges = N.arange( 0.0, 12.0001, binwidth )

n = 240
mat = 0.0
scales = N.geomspace( 1.0, 0.01, n )
for i, scale in zip(range(n), scales):
    mat += N.diag( N.full( n-i, scale ), i )
mat = N.asfortranarray( mat )

for eset in [
    [ [1.025], [3.025], [6.025], [9.025] ],
    [ [ 1.025, 5.025, 9.025 ] ],
    [ [ 6.025, 7.025,  8.025, 8.825 ] ],
    ]:
    ax = axes( 'Energy resolution impact' )
    for i, e in enumerate(eset):
        phist = singularities( e, edges )

        hist = R.Histogram( phist.size, edges, phist )
        esmear = R.EnergySmear( mat.shape[0], mat.ravel(), True )
        esmear.smear.Nvis( hist.hist )

        smeared = esmear.smear.Nrec.data()
        print( 'Sum check for {} (diff): {}'.format( e, phist.sum()-smeared.sum() ) )

        # bars = P.bar( edges[:-1], phist, binwidth, align='edge' )
        lines = plot_hist( edges, smeared )
        color = lines[0].get_color()
        ax.vlines( e, 0.0, smeared.max(), linestyle='--', color=color )

        if len(e)>1:
            color='green'
        for e in e:
            ax.plot( efine, binwidth*norm.pdf( efine, loc=e, scale=esmear.relativeSigma(e)*e ), linestyle='--', color=color )

    savefig( opts.output, suffix='_test_%i'%i )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'Synthetic energy leak matrix' )

mat = N.ma.array( mat, mask= mat==0.0 )
c = ax.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
add_colorbar( c )

savefig( opts.output, suffix='_mat' )

if opts.show:
    P.show()

