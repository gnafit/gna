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


mat = N.ones(100, dtype='d')

print( 'Input matrix (numpy)' )
print( mat )
print()

lmat = mat.ravel( order='F' )
shape = array_to_stdvector(mat.shape, 'size_t')

points = R.Points( lmat, shape )

identitygpu = R.IdentityGPU()
id2 = R.IdentityGPU()
identitygpu.identitygpu.source( points.points.points )
id2.identitygpu.source(identitygpu.identitygpu.target)
res = id2.identitygpu.target.data()

print( 'Result (C++ Data to numpy)' )
print( res )
print()

