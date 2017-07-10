#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the matrix memory ordering"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna import bindings
bindings.setup(R)
R.GNAObject

def array_to_stdvector( array, dtype ):
    """Convert an array to the std::vector<dtype>"""
    ret = R.vector(dtype)( len( array ) )
    for i, v in enumerate( array ):
        ret[i] = v
    return ret

#
# Create the matrix
#
mat = N.arange(12, dtype='d').reshape(3, 4)

# emat = R.Eigen.MatrixXd()

print( 'Input matrix (numpy)' )
print( mat )
print()

lmat = mat.ravel( order='F' )
shape = array_to_stdvector(mat.shape, 'size_t')

points = R.Points( lmat, shape )
identity = R.Identity()

identity.identity.source( points.points.points )

print( 'Eigen dump (C++)' )
identity.dump()
print()

res = identity.identity.target.data()

print( 'Result (C++ Data to numpy)' )
print( res )
print()

