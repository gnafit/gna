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
res = N.tril( res )

print( 'L' )
print( res )

diff = chol - res
print( 'Diff' )
print( diff )

print( ((diff<1.e-16).sum() and '\033[32mOK!' or '\033[31mFail!' ), '\033[0m' )

import IPython
IPython.embed()
