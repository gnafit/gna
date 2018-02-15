#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class

and test the matrix memory ordering"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from constructors import stdvector, Histogram
from gna.bindings import DataType

#
# Create the an array and bin edges
#
edges = N.arange(1.0, 7.1, 0.5)
arr = N.arange(edges.size-1)

print( 'Input edges and array (numpy)' )
print( edges )
print( arr )
print()

#
# Create transformations
#
hist = Histogram(edges, arr)
identity = R.Identity()
identity.identity.source(hist.hist.hist)

res = identity.identity.target.data()
dt  = identity.identity.target.datatype()

print( 'Eigen dump (C++)' )
identity.dump()
print()


print( 'Result (C++ Data to numpy)' )
print( res )
print()

print( 'Datatype:', str(dt) )
print( 'Edges:', list(dt.edges) )
