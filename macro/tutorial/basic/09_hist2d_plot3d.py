#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import constructors as C
import numpy as np
from gna.bindings import common
from matplotlib import pyplot as plt
# Create numpy array for data points
xmin, ymin = 10, 20
nbinsx, nbinsy = 40, 50
edgesx = np.linspace(xmin, xmin+nbinsx*0.5, nbinsx+1)**2
edgesy = np.linspace(ymin, ymin+nbinsy*0.5, nbinsy+1)**0.25
# Create fake data array
cx = (edgesx[1:] + edgesx[:-1])*0.5
cy = (edgesy[1:] + edgesy[:-1])*0.5
X, Y = np.meshgrid(cx, cy, indexing='ij')
narray = np.exp(-0.5*(X-cx[15])**2/150.0**2 - 0.5*(Y-cy[20])**2/0.10**2)

# Create a histogram instance with data, stored in `narray`
# and edges, sotred in `edges`
hist = C.Histogram2d(edgesx, edgesy, narray)

from mpl_tools.helpers import savefig
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '.png')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure()
ax = plt.subplot( 111, projection='3d' )
ax.set_title( 'surface' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_surface(cmap='viridis', colorbar=True)

savefig(oname, suffix='_surface')

fig = plt.figure()
ax = plt.subplot( 111, projection='3d' )
ax.set_title( 'bar3d' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_bar3d(cmap=True, colorbar=True)

savefig(oname, suffix='_bar3d')

fig = plt.figure()
ax = plt.subplot( 111, projection='3d' )
ax.set_title( 'bar3d' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_wireframe(cmap=True, colorbar=True)

savefig(oname, suffix='_wireframe')

plt.show()

