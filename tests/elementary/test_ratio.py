#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the Ratio transformation"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from constructors import Points

scale = 2.0
num   = 9
step  = 2
top    = N.arange(-num, num, step)
bottom = top*scale

top_o = Points(top)
bottom_o = Points(bottom)

ratio_o = R.Ratio(top_o, bottom_o)
result = ratio_o.ratio.ratio.data()

print('Scale', 1.0/scale)
print('Top', top)
print('Bottom', bottom)
print('Result (python)', top/bottom)
print('Result (GNA)', result)

check = N.allclose(result, 1.0/scale)
print( check and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )

# size = 4
# v = N.matrix(N.arange(size, dtype='d'))
# v.A1[size//2:] = N.arange(size//2, 0, -1)

# mat = v.T*v + N.eye( size, size )*size*2

# chol = N.linalg.cholesky( mat )

# print( 'Matrix (numpy)' )
# print( mat )
# print()

# print( 'L (numpy)' )
# print( chol )
# print()

# #
# # Create the transformations
# #
# points = Points( mat )
# cholesky = R.Cholesky()

# cholesky.cholesky.mat( points.points.points )

# #
# # Retrieve data
# #
# res = cholesky.cholesky.L.data()
# res = N.matrix(N.tril( res ))

# #
# # Print data
# #
# print( 'L' )
# print( res )

# mat_back = res*res.T

# print( 'Matrix (rec)' )
# print( mat_back )

# diff = chol - res
# print( 'Diff L' )
# print( diff )

# diff1 = mat_back - mat
# print( 'Diff mat' )
# print( diff1 )

# print( (((N.fabs(diff)+N.fabs(diff1))>1.e-12).sum() and '\033[31mFail!' or '\033[32mOK!' ), '\033[0m' )

