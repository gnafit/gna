#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test gpu"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator

def array_to_stdvector( array, dtype ):
    """Convert an array to the std::vector<dtype>"""
    ret = R.vector(dtype)( len( array ) )
    for i, v in enumerate( array ):
        ret[i] = v
    return ret


#
# Create the matrix
#
mat = N.arange(12, dtype='d')

# emat = R.Eigen.MatrixXd()

print( 'Input matrix (numpy)' )
print( mat )
print()

lmat = mat.ravel( order='F' )
shape = array_to_stdvector(mat.shape, 'size_t')

points = R.Points( lmat, shape )
print( 'Input 0' )
test = N.frombuffer( points.data(), count=mat.size )
print( test )

identity = R.IdentityX2()
id2 = R.IdentityGPU()
#idGPU = R.IdentityGPU()
identity.identityx2.source( points.points.points )
id2.identitygpu.source(identity.identityx2.target)
#print( 'Eigen dump (C++)' )
#identity.dump()
#print()

res = id2.identitygpu.target.data()

print( 'Input 1' )
test = N.frombuffer( points.data(), count=mat.size )
print( test )

print( 'Input 2' )
print(points.points.points.data())

print( 'Eigen dump (C++)' )
id2.dump()
print( 'Result (C++ Data to numpy)' )
print( res )
print()

