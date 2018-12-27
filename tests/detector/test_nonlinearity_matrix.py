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
from matplotlib import pyplot as P
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C

def rescale_to_matrix( edges_from, edges_to, **kwargs ):
    roundto = kwargs.pop( 'roundto', None )
    if not roundto is None:
        edges_from = N.round( edges_from, roundto )
        edges_to   = N.round( edges_to,   roundto )
    skipv = kwargs.pop( 'skip_values', [] )
    assert not kwargs

    idx = N.searchsorted( edges_from, edges_to, side='right' )-1
    dx = edges_to[1:]-edges_to[:-1]

    mat = N.zeros( shape=(edges_from.shape[0]-1, edges_from.shape[0]-1) )
    i1s = N.maximum( 0, idx[:-1] )
    i2s = N.minimum( idx[1:], edges_from.size-2 )
    for j, (i1, i2) in enumerate(zip(i1s, i2s)):
        if i2<0 or i1>=edges_from.size or edges_to[j]<-1.e100: continue
        for i in range( i1, i2+1 ):
            l1 = max( edges_to[j],   edges_from[i] )
            l2 = min( edges_to[j+1], edges_from[i+1] )
            w  = (l2-l1)/dx[j]
            mat[i,j] = w

    return mat

edges   = N.array( [   -1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ] )
edges_m = N.array( [ -2e100, -0.9, 0.5, 1.2, 1.8, 4.0, 5.0, 6.2, 7.5 ] )
matp = rescale_to_matrix( edges, edges_m, roundto=3 )

pedges_m = C.Points( edges_m )
ntrue = C.Histogram(edges, N.ones( edges.size-1 ) )

histedges = R.HistEdges()
histedges.histedges.hist( ntrue.hist )

nl = R.HistNonlinearity(True)
nl.set(histedges.histedges, pedges_m)
nl.add_input()


mat = nl.matrix.FakeMatrix.data()
print( 'C++' )
print( mat )
print( mat.sum( axis=0 ) )

print()
print( 'Python' )
print( matp )
print( matp.sum( axis=0 ) )
print()

diff = mat-matp
print( 'diff' )
print( diff )

print()
print( (diff==0.0).all() and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'Source bins' )
ax.set_ylabel( 'Target bins' )
ax.set_title( 'Bin edges scale conversion matrix' )

c = ax.matshow( N.ma.array(mat, mask=mat==0.0), extent=[edges[0], edges[-1], edges[-1], edges[0]] )
add_colorbar( c )

P.show()
