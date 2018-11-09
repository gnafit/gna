#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import constructors as C
import numpy as np
from gna.bindings import common

# Create numpy arrays for 1d and 2d cases
narray1 = np.arange(5, 10)
narray2 = np.arange(5, 20).reshape(5,3)
# Create a Points instances
parray1 = C.Points(narray1)
parray2 = C.Points(narray2)

print('Data 1d:',   parray1.points.points.data())
print('Data 2d:\n', parray2.points.points.data())

from matplotlib import pyplot as plt
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
ax.set_title( 'Plot title' )

parray1.points.points.plot('-o', label='plot 1d')
parray2.points.points.plot('-s', label='columns of 2d')

ax.legend(loc='upper left')

from mpl_tools.helpers import savefig
from sys import argv
oname = argv[0].rsplit('/', 1).pop().replace('.py', '.png')
savefig('output/tutorial/'+oname)

plt.show()
