#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C

parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
parser.add_argument( '-s', '--show', action='store_true' )
parser.add_argument( '-m', '--mode', default='upper', choices=[ 'upper', 'lower', 'both', 'none' ], help='which triangular part to fill' )
parser.add_argument( '-t', '--triangular', action='store_true', help='force transformation to account for upper triangular matrix' )
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

lower = 'lower' in opts.mode or 'both' in opts.mode
upper = 'upper' in opts.mode or 'both' in opts.mode
none = 'none' in opts.mode
n = 240
mat = 0.0
for i in range(n):
    if i<4:
        scale = 1.0 - i*0.2
    else:
        scale = 0.00005*(n-i)
    if i:
        if none:
            break
        if upper:
            mat += N.diag( N.full( n-i, scale ),  i )
        if lower:
            mat += N.diag( N.full( n-i, scale ), -i )
    else:
        mat += N.diag( N.full( n-i, scale ), i )
mat/=mat.sum( axis=0 )
pmat = C.Points( mat )

smtype = R.GNA.SquareMatrixType.UpperTriangular if opts.triangular else R.GNA.SquareMatrixType.Any
for eset in [
    [ [1.025], [3.025], [6.025], [9.025] ],
    [ [ 1.025, 5.025, 9.025 ] ],
    [ [ 6.025, 7.025,  8.025, 8.825 ] ],
    ]:
    ax = axes( 'Energy leak impact' )
    for i, e in enumerate(eset):
        phist = singularities( e, edges )

        hist = C.Histogram( edges, phist )
        esmear = R.HistSmear(smtype)
        esmear.smear.inputs.SmearMatrix( pmat.points )
        esmear.smear.inputs.Ntrue( hist.hist )

        smeared = esmear.smear.Nrec.data()
        print( 'Sum check for {} (diff): {}'.format( e, phist.sum()-smeared.sum() ) )

        # bars = P.bar( edges[:-1], phist, binwidth, align='edge' )
        lines = plot_hist( edges, smeared )
        color = lines[0].get_color()
        ax.vlines( e, 0.0, smeared.max(), linestyle='--', color=color )

    savefig( opts.output, suffix='_test_%i'%i )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'Synthetic energy leak matrix' )

mat = pmat.points.points.data()
mat = N.ma.array( mat, mask= mat==0.0 )
c = ax.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
add_colorbar( c )

savefig( opts.output, suffix='_mat' )

if opts.show:
    P.show()

