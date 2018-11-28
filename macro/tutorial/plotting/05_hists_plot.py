#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna import tutorial
import constructors as C
import numpy as np
from gna.bindings import common
from matplotlib import pyplot as plt
# Create numpy array for data points
nbins = 40
# Create numpy array for bin edges
edges  = np.linspace(1.0, 10.0, nbins+1)
narray1 = np.exp(edges[:-1])
narray2 = np.flip(narray1)
narray3 = narray1[-1]*np.exp(-0.5*(5.0-edges[:-1])**2/1.0)/(2*np.pi)**0.5

# Create a histogram instance with data, stored in `narray`
# and edges, sotred in `edges`
hist1 = C.Histogram(edges, narray1)
hist2 = C.Histogram(edges, narray2)
hist3 = C.Histogram(edges, narray3)

from mpl_tools.helpers import savefig
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '.png')

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
plt.ticklabel_format(style='sci', axis='y', scilimits=(2,2))

hist1.hist.hist.plot_hist(label='exp(+)')
hist2.hist.hist.plot_hist(label='exp(-)')
hist3.hist.hist.plot_hist(label='gauss')

ax.legend()
savefig(oname, suffix='_hist')

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
plt.ticklabel_format(style='sci', axis='y', scilimits=(2,2))

hist1.hist.hist.plot_bar(label='exp(+)', alpha=0.4)
hist2.hist.hist.plot_bar(label='exp(-)', alpha=0.4)
hist3.hist.hist.plot_bar(label='gauss' , alpha=0.4)

ax.legend()
savefig(oname, suffix='_bar1')

fig = plt.figure()
ax = plt.subplot( 111 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
plt.ticklabel_format(style='sci', axis='y', scilimits=(2,2))

hist1.hist.hist.plot_bar(label='exp(+)', divide=3, shift=0)
hist2.hist.hist.plot_bar(label='exp(-)', divide=3, shift=1)
hist3.hist.hist.plot_bar(label='gauss' , divide=3, shift=2)

ax.legend()
savefig(oname, suffix='_bar2')

plt.show()
