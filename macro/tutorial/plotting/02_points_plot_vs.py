#!/usr/bin/env python

from tutorial import tutorial_image_name, savefig
import gna.constructors as C
import numpy as np
from gna.bindings import common

# Create numpy arrays for 1d and 2d cases
narray_x = np.arange(5, 10)
narray_y = np.arange(0.13, 0.08, -0.01)
# Create Points instances
parray_x = C.Points(narray_x)
parray_y = C.Points(narray_y)

from matplotlib import pyplot as plt
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
ax.set_title( 'Plot title' )

parray_y.points.points.plot_vs(parray_x.points.points, '-o', label='plot Y vs X')

ax.legend(loc='upper right')

savefig(tutorial_image_name('png'))

plt.show()
