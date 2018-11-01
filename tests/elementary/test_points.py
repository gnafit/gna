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
mat = N.arange(12, dtype='d').reshape(3, 4)

print( 'Input matrix (numpy)' )
print( mat )
print()

#
# Create transformations
#
points = Points(mat)
identity = R.Identity()

identity.identity.source( points.points.points )
res = identity.identity.target.data()
dt  = identity.identity.target.datatype()

#
# Dump
#
print( 'Eigen dump (C++)' )
identity.dump()
print()

print( 'Result (C++ Data to numpy)' )
print( res )
print()

print( 'Datatype:', str(dt) )

