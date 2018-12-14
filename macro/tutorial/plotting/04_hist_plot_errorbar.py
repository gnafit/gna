#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna import tutorial
import constructors as C
import numpy as np
from gna.bindings import common
from matplotlib import pyplot as plt
# Create numpy array for data points
nbins = 10
narray = np.arange(nbins)**2 * np.arange(nbins)[::-1]**2
yerr   = np.ones_like(narray)*20
# Create numpy array for bin edges
edges  = np.linspace(1.0, 40.0, nbins+1)

# Create a histogram instance with data, stored in `narray`
# and edges, sotred in `edges`
hist = C.Histogram(edges, narray)

fig = plt.figure(figsize=(12, 5))
ax = plt.subplot( 121 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_errorbar(yerr, label='with input errors')

ax.legend(loc='lower center')

ax = plt.subplot( 122 )
ax.set_title( 'Plot title (right)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )

hist.hist.hist.plot_errorbar(yerr='stat', label='with stat errors')

ax.legend(loc='lower center')

from mpl_tools.helpers import savefig
from sys import argv
oname = argv[0].rsplit('/', 1).pop().replace('.py', '.png')
savefig('output/tutorial/'+oname)

plt.show()
