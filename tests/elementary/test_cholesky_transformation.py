#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the Cholesky transformation"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna import bindings
bindings.setup(R)
# R.GNAObject

def array_to_stdvector( array, dtype ):
    """Convert an array to the std::vector<dtype>"""
    ret = R.vector(dtype)( len( array ) )
    for i, v in enumerate( array ):
        ret[i] = v
    return ret

#
# Create the matrix
#
size = 4
v = N.matrix(N.arange(size, dtype='d'))
v.A1[size//2:] = N.arange(size//2, 0, -1)

mat = v.T*v + N.eye( size, size )*size*2

chol = N.linalg.cholesky( mat )

print( 'Matrix (numpy)' )
print( mat )
print()

print( 'L (numpy)' )
print( chol )
print()

lmat = mat.A.ravel( order='F' )
shape = array_to_stdvector(mat.shape, 'size_t')

points = R.Points( lmat, shape )
cholesky = R.Cholesky()

cholesky.cholesky.mat( points.points.points )

res = cholesky.cholesky.L.data()
res = N.matrix(N.tril( res ))

print( 'L' )
print( res )

mat_back = res*res.T

print( 'Matrix (rec)' )
print( mat_back )

diff = chol - res
print( 'Diff L' )
print( diff )

diff1 = mat_back - mat
print( 'Diff mat' )
print( diff1 )

print( (((N.fabs(diff)+N.fabs(diff1))>1.e-12).sum() and '\033[31mFail!' or '\033[32mOK!' ), '\033[0m' )

