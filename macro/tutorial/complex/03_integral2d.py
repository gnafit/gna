#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.env import env
import constructors as C
import numpy as np
from matplotlib import pyplot as plt
from gna.bindings import common
import ROOT as R
from mpl_toolkits.mplot3d import Axes3D

# Make a variable for global namespace
ns = env.globalns

# Create a parameter in the global namespace
pa = ns.defparameter('a', central=1.0, free=True, label='x scale')
pb = ns.defparameter('b', central=2.0, free=True, label='y scale')

# Print the list of parameters
ns.printparameters(labels=True)
print()

# Define binning and integration orders
x_nbins = 20
x_edges  = np.linspace(-np.pi, np.pi, x_nbins+1, dtype='d')
x_widths = x_edges[1:]-x_edges[:-1]
x_orders = 4

y_nbins = 30
y_edges  = np.linspace(0, 2.0*np.pi, y_nbins+1, dtype='d')
y_widths = y_edges[1:]-y_edges[:-1]
y_orders = 3

# Initialize histogram
hist = C.Histogram2d(x_edges, y_edges)

# Initialize integrator
integrator = R.Integrator2GL(x_nbins, x_orders, y_nbins, y_orders)
integrator.points.edges(hist.hist.hist)
int_points = integrator.points

# Create integrable: a*sin(x) + b*cos(k*x)
arg_t = C.WeightedSum( ['a', 'b'], [int_points.xmesh, int_points.ymesh] )
sin_t = R.Sin(arg_t.sum.sum)

# integrator.add_input(sint_t.sin.result)
integrator.hist.f(sin_t.sin.result)
X, Y = integrator.points.xmesh.data(), integrator.points.ymesh.data()

integrator.print()
print()

# Label transformations
hist.hist.setLabel('Input histogram\n(bins definition)')
integrator.points.setLabel('Sampler\n(Gauss-Legendre)')
integrator.hist.setLabel('Integrator\n(convolution)')
sin_t.sin.setLabel('sin(ax+by)')
arg_t.sum.setLabel('ax+by')

# Make 2d color plot
fig = plt.figure()
ax = plt.subplot(111, xlabel='x', ylabel='y', title=r'$\int\int\sin(ax+by)$')
ax.minorticks_on()
ax.set_aspect('equal')

# Draw the function and integrals
integrator.hist.hist.plot_pcolormesh(colorbar=True)

# Save figure
from mpl_tools.helpers import savefig
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '')
savefig(oname+'.png')

# Add integration points and save
ax.scatter(X, Y, c='red', marker='.', s=0.2)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0.0, 1.0)

savefig(oname+'_zoom.png')

# Plot 3d function and a histogram
fig = plt.figure()
ax = plt.subplot(111, xlabel='x', ylabel='y', title=r'$\sin(ax+by)$', projection='3d')
ax.minorticks_on()
ax.view_init(elev=17., azim=-33)

# Draw the function and integrals
integrator.hist.hist.plot_surface(cmap='viridis', colorbar=True)
sin_t.sin.result.plot_wireframe_vs(X, Y, rstride=8, cstride=8)

savefig(oname+'_3d.png')

# Save the graph
from gna.graphviz import savegraph
savegraph(sin_t.sin, oname+'_graph.png')

plt.show()
