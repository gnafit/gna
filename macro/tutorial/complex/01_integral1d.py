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
nbins = 30
x_edges = np.linspace(-1.0*np.pi, 1.0*np.pi, nbins+1, dtype='d')
orders = 3

# Initialize integrator
integrator = R.IntegratorGL(nbins, orders, x_edges)
int_points = integrator.points.x

# Create integrable: a*sin(x) + b*cos(k*x)
cos_arg = C.WeightedSum( ['k'], [int_points] )
sin_t = R.Sin(int_points)
cos_t = R.Cos(cos_arg.sum.sum)
fcn   = C.WeightedSum(['a', 'b'], [sin_t.sin.result, cos_t.cos.result])

integrator.hist.f(fcn.sum.sum)

# Print objects
integrator.print()
print()

fcn.print()
print()

cos_t.print()
print()

# Label transformations
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
gs = gridspec.GridSpec(nrows=4, ncols=1, hspace=0.0)
ax = plt.subplot(gs[:3,0])
ax.minorticks_on()
# ax.grid()
ax.set_ylabel( 'f(x)' )
ax.set_title(r'$a\,\sin(x)+b\,\sin(kx)$')

# Draw the function and integrals
fcn.sum.sum.plot_vs(int_points, 'o-', label='function at integration points', alpha=0.5, markerfacecolor='none', markersize=4.0)
integrator.hist.hist.plot_bar(label='integral: numerical', divide=2, shift=0)
ax.bar(bar_left, integral_a, bar_width, label='integral: analytical', align='edge')

# Freeze axis limits and draw bin edges
ax.autoscale(enable=False, axis='y')
ymin, ymax = ax.get_ylim()
ax.vlines(integrator.points.xedges.data(), ymin, ymax, linestyle='--', alpha=0.4, linewidth=0.5)

ax.legend(loc='lower right')

# Add difference
ax = plt.subplot(gs[3,0], sharex=ax)
ax.set_xlabel('x')
diff_factor=1.e-7
ax.set_ylabel(r'diff., $\times10^{-7}$')

diff = integral_n-integral_a
ax.bar(x_edges[:-1], diff/diff_factor, bar_width*2.0, align='edge')

# Freeze axis limits and draw bin edges
ax.autoscale(enable=False, axis='y')
ymin, ymax = ax.get_ylim()
ax.vlines(integrator.points.xedges.data(), ymin, ymax, linestyle='--', alpha=0.4, linewidth=0.5)

# Save figure and graph as images
savefig(tutorial_image_name('png'))

savegraph(fcn.sum, tutorial_image_name('png', suffix='graph'), rankdir='TB')

plt.show()
