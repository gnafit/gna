#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class

and test the matrix memory ordering"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from constructors import stdvector, Points
from gna.bindings import DataType

#
# Create the matrix
#
mat = N.arange(5, dtype='d')

print( 'Input array (numpy)' )
print( mat )
print()

#
# Create transformations
#
print('Configuration')
points = Points(mat)
multifunc = R.TrialMultiFunc()
trans = multifunc.multifunc
trans.inp(points)
out = trans.out

print()
print('Run in main mode')
print('    result', out.data())

print()
print('Switch to secondary mode')
trans.switchFunction('secondary')
trans.taint()
print('    result', out.data())

print()
print('Switch to secondaryMem mode')
trans.switchFunction('secondaryMem')
trans.taint()
print('    result', out.data())

print()
print('Switch to thirdparty mode')
trans.switchFunction('thirdparty')
trans.taint()
print('    result', out.data())

print()
print('Switch to main mode again')
trans.switchFunction('main')
trans.taint()
print('    result', out.data())

# print()
# print('Try to switch to non existing mode')
# trans.switchFunction('this mode does not exist')
