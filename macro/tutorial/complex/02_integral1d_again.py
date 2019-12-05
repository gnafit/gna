#!/usr/bin/env python

from __future__ import print_function
from tutorial import tutorial_image_name, savefig, savegraph
import load
from gna.env import env
import gna.constructors as C
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from gna.bindings import common
import ROOT as R

# Make a variable for global namespace
ns = env.globalns

# Create a parameter in the global namespace
pa = ns.defparameter('a', central=1.0, free=True, label='weight 1')
pb = ns.defparameter('b', central=0.2, free=True, label='weight 2')
pk = ns.defparameter('k', central=8.0, free=True, label='argument scale')

# Print the list of parameters
ns.printparameters(labels=True)
print()

# Define binning and integration orders
nbins = 20
x_edges = -np.pi + 2.0*np.pi*np.linspace(0.0, 1.0, nbins+1, dtype='d')**0.6
x_widths = x_edges[1:]-x_edges[:-1]
x_width_ratio = x_widths/x_widths.min()
orders = np.ceil(x_width_ratio*4).astype('i')

# Initialize histogram
hist = C.Histogram(x_edges)

# Initialize integrator
integrator = R.IntegratorTrap(nbins, orders)
integrator.points.edges(hist.hist.hist)
int_points = integrator.points.x

# Create integrable: a*sin(x) + b*cos(k*x)
cos_arg = C.WeightedSum( ['k'], [int_points] )
sin_t = R.Sin(int_points)
cos_t = R.Cos(cos_arg.sum.sum)
fcn   = C.WeightedSum(['a', 'b'], [sin_t.sin.result, cos_t.cos.result])

integrator.add_input(fcn.sum.sum)

integrator.print()
print()

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



savefig(tutorial_image_name('png', suffix='1'))

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

savefig(tutorial_image_name('png', suffix='2'))


savegraph(fcn.sum, tutorial_image_name('png', suffix='graph'))

plt.show()
