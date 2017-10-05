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
        if i2<0 or i1>=edges_from.size: continue
        for i in range( i1, i2+1 ):
            l1 = max( edges_to[j],   edges_from[i] )
            l2 = min( edges_to[j+1], edges_from[i+1] )
            w  = (l2-l1)/dx[j]
            mat[i,j] = w

    return mat

edges   = N.array( [  0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ] )
edges_m = N.array( [ -1.0, 0.5, 1.2, 1.8, 4.0, 5.0, 6.2, 7.5 ] )
matp = rescale_to_matrix( edges_m, edges, roundto=3 )

pedges, pedges_m = convert( edges, 'points' ), convert( edges_m, 'points' )
ntrue = R.Histogram( edges.size-1, edges, N.ones( edges.size-1 ) )

nl = R.EnergyNonlinearity()
nl.set( pedges, pedges_m, ntrue )

idy = R.Identity()
idy.identity.source(nl.matrix.Matrix)

mat = idy.identity.target.data()
print( 'C++' )
print( mat )
print( mat.sum( axis=0 ) )

print()
print( 'Python' )
print( matp )
print( matp.sum( axis=0 ) )
print()

print( 'diff' )
print( mat-matp )

import IPython
IPython.embed()
