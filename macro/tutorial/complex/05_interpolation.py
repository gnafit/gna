#!/usr/bin/env python

from __future__ import print_function
from tutorial import tutorial_image_name, savefig, savegraph
from load import ROOT as R
import gna.constructors as C
import numpy as np
from matplotlib import pyplot as plt
from gna.bindings import common
from gna.env import env

# Initialize segment edges and sampling points
xmin, xmax = 1.0, 11.0
nsegments = 10
npoints   = 100
edges = C.Points(np.linspace(xmin, xmax, nsegments+1, dtype='d'), labels='Coarse x')
edges_data = edges.single().data()

points = np.linspace(xmin, xmax, npoints+1, dtype='d')
np.random.shuffle(points)
points   = C.Points(points, labels='Fine x\n(shuffled)')
# Note: sampling points are shuffled

# Initialize raw function
y0 = np.exp(1/edges.single().data()**0.5)
# Make it saw-like by scaling all odd and even points to different sides
y0[::2]*=1.4
y0[1::2]*=0.7
y0 = C.Points(y0, labels='Coarse y\n(not scaled)')

# Define two sets of scales to correct each value of y0
pars1 = env.globalns('pars1')
pars2 = env.globalns('pars2')
for ns in (pars1, pars2):
    for i in range(nsegments+1):
        ns.defparameter('par_{:02d}'.format(i), central=1, free=True, label='Scale for x_{}={}'.format(i, edges_data[i]))

# Initialize transformations for scales
with pars1:
    varray1=C.VarArray(pars1.keys(), labels='Scales 1\n(p1)')

with pars2:
    varray2=C.VarArray(pars2.keys(), labels='Scales 2\n(p2)')

# Make two products: y0 scaled by varray1 and varray2
y1 = C.Product([varray1.single(), y0.single()], labels='p1*y')
y2 = C.Product([varray2.single(), y0.single()], labels='p2*y')

# Initialize interpolator
manual = False
labels=('Segment index\n(fine x in coarse x)', 'Interpolator')
if manual:
    # Bind transformations manually
    interpolator = R.InterpExpo(labels=labels)

    edges  >> (interpolator.insegment.edges, interpolator.interp.x)
    points >> (interpolator.insegment.points, interpolator.interp.newx)
    interpolator.insegment.insegment >> interpolator.interp.insegment
    interpolator.insegment.widths    >> interpolator.interp.widths
    y1 >> interpolator.interp.y
    y2 >> interpolator.add_input()

    interp1, interp2 = interpolator.interp.outputs.values()
else:
    # Bind transformations automatically
    interpolator = R.InterpExpo(edges, points, labels=labels)
    interp1 = interpolator.add_input(y1)
    interp2 = interpolator.add_input(y2)

# Print the interpolator status
interpolator.print()

# Change each second parameter for second curve
for par in pars2.values()[1::2]:
    par.set(2.0)

# Print parameters
env.globalns.printparameters(labels=True)

# Plot graphs
fig = plt.figure()
ax = plt.subplot(111, xlabel='x', ylabel='y', title='Interpolation')
ax.minorticks_on()

y0.points.points.plot_vs(edges.single(), '--', label='Original function')
interp1.plot_vs(points.single(), 's', label='Interpolation (scale 1)', markerfacecolor='none')
interp2.plot_vs(points.single(), '^', label='Interpolation (scale 2)', markerfacecolor='none')

for par in pars1.values()[::2]:
    par.set(0.50)

interp1.plot_vs(points.single(), 'v', label='Interpolation (scale 1, modified)', markerfacecolor='none')

ax.autoscale(enable=False, axis='y')
ymin, ymax = ax.get_ylim()
ax.vlines(edges_data, ymin, ymax, linestyle='--', label='Segment edges', alpha=0.5, linewidth=0.5)

ax.legend(loc='upper right')

# Save figure and graph as images



savefig(tutorial_image_name('png'))

savegraph(points, tutorial_image_name('png', suffix='graph'))

plt.show()

