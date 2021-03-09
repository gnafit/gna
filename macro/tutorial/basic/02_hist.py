#!/usr/bin/env python

import gna.constructors as C
import numpy as np
# Create numpy array for data points
nbins = 12
narray = np.arange(nbins)**2 * np.arange(nbins)[::-1]**2
# Create numpy array for bin edges
edges  = np.linspace(1.0, 7.0, nbins+1)

# Create a histogram instance with data, stored in `narray`
# and edges, stored in `edges`
hist = C.Histogram(edges, narray)
hist.print()
print()

# Access the output `points` of transformation `points` of the object `parray`
print('Output:', hist.hist.hist)
# Access and print relevant DataType
datatype = hist.hist.hist.datatype()
print('DataType:', datatype)
print('Bin edges:', list(datatype.edges))
# Access the actual data
print('Data:', hist.hist.hist.data())
