#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import constructors as C
import numpy as np
# Create numpy arrays for bin edges
nbinsx, nbinsy = 12, 8
edgesx = np.linspace(0, nbinsx, nbinsx+1)
edgesy = np.linspace(0, nbinsy, nbinsy+1)
# Create fake data array
narray = np.arange(nbinsx*nbinsy).reshape(nbinsx, nbinsy)
narray = narray**2 * narray[::-1,::-1]**2

# Create a histogram instance with data, stored in `narray`
# and edges, sotred in `edgesx` and `edgesy`
hist = C.Histogram2d(edgesx, edgesy, narray)

# Import helper library to make print output more informative
from gna.bindings import common
# Access the output `points` of transformation `points` of the object `parray`
print('Output:', hist.hist.hist)
# Access and print relevant DataType
datatype = hist.hist.hist.datatype()
print('DataType:', datatype)
print('Bin edges (X):', list(datatype.edgesNd[0]))
print('Bin edges (Y):', list(datatype.edgesNd[1]))
# Access the actual data
print('Data:', hist.hist.hist.data())
