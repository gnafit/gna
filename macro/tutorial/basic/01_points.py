#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import gna.constructors as C
import numpy as np
# Create numpy array
narray = np.arange(12).reshape(3,4)
# Create a points instance with data, stored in `narray`
parray = C.Points(narray)

# Print the structure of GNAObject parray
parray.print()
print()

# Print list of transformations
print('Transformations:', parray.transformations.keys())

# Print list of outputs
print('Outputs:', parray.points.outputs.keys())
print()

# Access the output `points` of transformation `points` of the object `parray`
print('Output:', parray.points.points)
# Access and print relevant DataType
print('DataType:', parray.points.points.datatype())
# Access the actual data
print('Data:\n', parray.points.points.data())
