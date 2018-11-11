#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import constructors as C
import numpy as np
from gna.bindings import common
from matplotlib import pyplot as plt
# Create numpy array for data points
nbinsx, nbinsy = 40, 50
edgesx = np.linspace(0, nbinsx, nbinsx+1)
edgesy = np.linspace(0, nbinsy, nbinsy+1)
# Create fake data array
cx = (edgesx[1:] + edgesx[:-1])*0.5
cy = (edgesy[1:] + edgesy[:-1])*0.5
X, Y = np.meshgrid(cx, cy, indexing='ij')
# narray = np.exp(-0.5*(X-20.0)**2/10.0**2 - 0.5*(Y-25.0)**2/5.0**2 - 0.5*(X-20.0)*(Y-25.0)/10.0/5.0)
narray = np.exp(-0.5*(X-20.0)**2/15.0**2 - 0.5*(Y-25.0)**2/3.0**2)

# Create a histogram instance with data, stored in `narray`
# and edges, sotred in `edges`
hist = C.Histogram2d(edgesx, edgesy, narray)

from mpl_tools.helpers import savefig
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '.png')

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_pcolorfast(mask=0.0, colorbar=True)

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_pcolormesh(mask=0.0, colorbar=True)

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_pcolor(mask=0.0, colorbar=True)

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_imshow(mask=0.0, colorbar=True)

savefig(oname)

plt.show()
