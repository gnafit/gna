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
parser.add_argument( '-u', '--upper', action='store_true', help='force transformation to account for upper triangular matrix' )
parser.add_argument( '-O', '--offdiag', action='store_true', help='force transformation to account for upper triangular matrix' )
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

par = env.defparameter( 'DiagScale',  central=1.0, relsigma=0.1 )
binwidth=0.05
edges = N.arange( 0.0, 12.0001, binwidth )

n = 240
mat = 0.0
for i in range(n):
    if i<4:
        scale = 1.0 - i*0.2
    else:
        scale = 0.00005*(n-i)
    if i:
        mat += N.diag( N.full( n-i, scale ),  i )
        if not opts.upper:
            mat += N.diag( N.full( n-i, scale ), -i )
    else:
        mat += N.diag( N.full( n-i, scale ), i )
mat/=mat.sum( axis=0 )
pmat = C.Points( mat )

ax = axes( 'Energy leak impact' )
phist = singularities( [ 1.025, 5.025, 9.025 ], edges )
hist = C.Histogram( edges, phist )

ndiag = 4
rd = R.RenormalizeDiag( ndiag, int(opts.offdiag), int(opts.upper) )
rd.renorm.inmat( pmat.points )

esmear = R.HistSmear( opts.upper )
esmear.smear.inputs.SmearMatrix( rd.renorm )
esmear.smear.inputs.Ntrue( hist.hist )

for i, value in enumerate([ 1.0, 0.5, 2.0 ]):
    par.set( value )
    smeared = esmear.smear.Nrec.data()
    print( 'Sum check for {} (diff): {}'.format( value, phist.sum()-smeared.sum() ) )

    # bars = P.bar( edges[:-1], phist, binwidth, align='edge' )
    P.sca( ax )
    lines = plot_hist( edges, smeared, label='%.2f'%value )
    color = lines[0].get_color()

    fig = P.figure()
    ax1 = P.subplot( 111 )
    ax1.minorticks_on()
    ax1.grid()
    ax1.set_xlabel( '' )
    ax1.set_ylabel( '' )
    ax1.set_title( 'Synthetic energy leak matrix (diag scale=%.2f, ndiag=%i)'%(value, ndiag) )

    mat = rd.renorm.outmat.data()
    mat = N.ma.array( mat, mask= mat==0.0 )
    c = ax1.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
    add_colorbar( c )

    savefig( opts.output, suffix='_mat_%i'%i )

ax.legend()
savefig( opts.output, suffix='_hist' )


if opts.show:
    P.show()

