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
x_edges  = -np.pi + 2.0*np.pi*np.linspace(0.0, 1.0, x_nbins+1, dtype='d')**0.6
x_widths = x_edges[1:]-x_edges[:-1]
x_orders = np.ceil(x_widths/x_widths.min()*4).astype('i')

y_nbins = 30
y_edges  = 0.0    + 2.0*np.pi*np.linspace(0.0, 1.0, y_nbins+1, dtype='d')**1.4
y_widths = y_edges[1:]-y_edges[:-1]
y_orders = np.ceil(y_widths/y_widths.min()*4).astype('i')

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

integrator.print()
print()

from sys import exit
exit(1)

# Label transformations
hist.hist.setLabel('Input histogram\n(bins definition)')
integrator.points.setLabel('Sampler\n(Gauss-Legendre)')
integrator.hist.setLabel('Integrator\n(convolution)')
cos_arg.sum.setLabel('kx')
sin_t.sin.setLabel('sin(x)')
cos_t.cos.setLabel('cos(kx)')
fcn.sum.setLabel('a sin(x) + b cos(kx)')

# Compute the integral analytically as a cross check
# int| a*sin(x) + b*cos(k*x) = a*cos(x) - b/k*sin(kx) + C
a, b, k = (p.value() for p in (pa, pb, pk))
F = - a*np.cos(x_edges) + (b/k)*np.sin(k*x_edges)
integral_a = F[1:] - F[:-1]
bar_width = (x_edges[1:]-x_edges[:-1])*0.5
bar_left = x_edges[:-1]+bar_width

integral_n = integrator.hist.hist.data()

# Do some plotting
fig = plt.figure()
ax = plt.subplot(111)
ax.minorticks_on()
# ax.grid()
ax.set_ylabel( 'f(x)' )
ax.set_title(r'$a\,\sin(x)+b\,\sin(kx)$')

# Draw the function and integrals
fcn.sum.sum.plot_vs(int_points, 'o-', label='function at integration points', alpha=0.5, markerfacecolor='none', markersize=4.0)
integrator.hist.hist.plot_bar(label='integral: numerical', divide=2, shift=0, alpha=0.6)
ax.bar(bar_left, integral_a, bar_width, label='integral: analytical', align='edge', alpha=0.6)

# Freeze axis limits and draw bin edges
ax.autoscale(enable=False, axis='y')
ymin, ymax = ax.get_ylim()
ax.vlines(integrator.points.xedges.data(), ymin, ymax, linestyle='--', alpha=0.4, linewidth=0.5)

ax.legend(loc='lower right')

ymin, ymax = ax.get_ylim()

# Save figure and graph as images
from mpl_tools.helpers import savefig
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '')
savefig(oname+'_1.png')

# Do more plotting
fig = plt.figure()
ax = plt.subplot(111)
ax.minorticks_on()
# ax.grid()
ax.set_ylabel( 'f(x)' )
ax.set_title(r'$a\,\sin(x)+b\,\sin(kx)$')
ax.autoscale(enable=False, axis='y')
ax.set_ylim(ymin, ymax)
ax.axhline(0.0, linestyle='--', color='black', linewidth=1.0, alpha=0.5)

def plot_sample():
    label = 'a={}, b={}, k={}'.format(*(p.value() for p in (pa,pb,pk)))
    lines = fcn.sum.sum.plot_vs(int_points, '--', alpha=0.5, linewidth=1.0)
    color = lines[0].get_color()
    integrator.hist.hist.plot_hist(alpha=0.6, color=color, label=label)

plot_sample()
pa.set(-0.5)
pk.set(16)
plot_sample()
pa.set(0)
pb.set(0.1)
plot_sample()


# Freeze axis limits and draw bin edges
ymin, ymax = ax.get_ylim()
ax.vlines(integrator.points.xedges.data(), ymin, ymax, linestyle='--', alpha=0.4, linewidth=0.5)

ax.legend(loc='lower right')

savefig(oname+'_2.png')

from gna.graphviz import savegraph
savegraph(fcn.sum, oname+'_graph.png')

plt.show()
