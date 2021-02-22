#!/usr/bin/env python

from tutorial import tutorial_image_name
import gna.constructors as C
import numpy as np
from gna.bindings import common
from matplotlib import pyplot as plt
# Create numpy array for data points
nbins = 100
narray = np.arange(nbins)**2 * np.arange(nbins)[::-1]**2
# Create numpy array for bin edges
edges  = np.linspace(1.0, 40.0, nbins+1)

# Create a histogram instance with data, stored in `narray`
# and edges, stored in `edges`
hist = C.Histogram(edges, narray)

fig = plt.figure(figsize=(12, 5))
ax = plt.subplot( 121 )
ax.set_title( 'Plot title (left)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)

hist.hist.hist.plot_hist(label='histogram 1')

ax.legend()

ax = plt.subplot( 122 )
ax.set_title( 'Plot title (right)' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x label' )
ax.set_ylabel( 'y label' )
plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)

hist.hist.hist.plot_bar(label='histogram 1 (bar)', alpha=0.6)

ax.legend()

from mpl_tools.helpers import savefig

savefig(tutorial_image_name('png'))

plt.show()
