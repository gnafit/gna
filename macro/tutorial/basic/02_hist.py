#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import constructors as C
import numpy as N
# Create numpy array for data points
nbins = 12
narray = N.arange(nbins)**2 * N.arange(nbins)[::-1]**2
# Create numpy array for bin edges
edges  = N.linspace(1.0, 7.0, nbins+1)

# Create a histogram instance with data, stored in `narray`
# and edges, sotred in `edges`
hist = C.Histogram(edges, narray)

# Import helper library to make print output more informative
from gna import printing
# Access the output `points` of transformation `points` of the object `parray`
print('Output:', hist.hist.hist)
# Access and print relevant DataType
datatype = hist.hist.hist.datatype()
print('DataType:', datatype)
print('Bin edges:', list(datatype.edges))
# Access the actual data
print('Data:', hist.hist.hist.data())
