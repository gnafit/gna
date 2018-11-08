#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import constructors as C
import numpy as np
# Create numpy array
narray = np.arange(12)
# Create a points instance with data, stored in `narray`
parray = C.Points(narray)

# Import helper library to make print output more informative
from gna import printing
# Access the output `points` of transformation `points` of the object `parray`
print('Output:', parray.points.points)
# Access and print relevant DataType
print('DataType:', parray.points.points.datatype())
# Access the actual data
print('Data:\n', parray.points.points.data())

from matplotlib import pyplot as plt
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
ax.set_title( 'Plot title' )

parray.points.points.plot('-o', label='plot 1')

ax.legend(loc='upper left')
plt.show()

