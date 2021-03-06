#!/usr/bin/env python

from tutorial import tutorial_image_name, savefig
import gna.constructors as C
import numpy as np
from gna.bindings import common
from matplotlib import pyplot as plt
# Create numpy array for data points
xmin, ymin = 10, 20
nbinsx, nbinsy = 40, 50
edgesx = np.linspace(xmin, xmin+nbinsx*0.5, nbinsx+1)
edgesy = np.linspace(ymin, ymin+nbinsy*0.5, nbinsy+1)
# Create fake data array
cx = (edgesx[1:] + edgesx[:-1])*0.5
cy = (edgesy[1:] + edgesy[:-1])*0.5
X, Y = np.meshgrid(cx, cy, indexing='ij')
narray = np.exp(-0.5*(X-15.0)**2/10.0**2 - 0.5*(Y-30.0)**2/3.0**2)

# Create a histogram instance with data, stored in `narray`
# and edges, stored in `edges`
hist = C.Histogram2d(edgesx, edgesy, narray)

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'pcolorfast' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_pcolorfast(colorbar=True)

savefig(tutorial_image_name('png', suffix='pcolorfast'))

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'imshow' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_imshow(colorbar=True)

savefig(tutorial_image_name('png', suffix='imshow'))

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'matshow' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_matshow(colorbar=True)

savefig(tutorial_image_name('png', suffix='matshow'))

plt.show()
