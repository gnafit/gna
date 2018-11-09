#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import constructors as C
import numpy as np
from gna.bindings import common

# Create numpy arrays for 1d and 2d cases
narray2 = np.arange(20).reshape(5,4)
# Create a Points instances
parray2 = C.Points(narray2)

print('Data 2d:\n', parray2.points.points.data())

from matplotlib import pyplot as plt
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
ax.set_title( 'Plot title' )

parray2.points.points.matshow(colorbar=True, mask=0.0)

from mpl_tools.helpers import savefig
from sys import argv
oname = argv[0].rsplit('/', 1).pop().replace('.py', '.png')
savefig('output/tutorial/'+oname)

plt.show()
