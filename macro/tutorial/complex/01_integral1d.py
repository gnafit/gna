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
p1 = ns.defparameter('a', central=1.0, free=True, label='weight 1')
p2 = ns.defparameter('b', central=0.2, free=True, label='weight 2')
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
# integrator.points.x.setLabel('x')
# integrator.points.xedges.setLabel('          bin edges')
integrator.hist.setLabel('Integrator\n(convolution)')
cos_arg.sum.setLabel('kx')
sin_t.sin.setLabel('sin(x)')
cos_t.cos.setLabel('cos(kx)')
fcn.sum.setLabel('a sin(x) + b cos(kx)')

# Do some plotting
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
# ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'f(x)' )
ax.set_title(r'$a\,\sin(x)+b\,\sin(kx)$')

fcn.sum.sum.plot_vs(int_points, 'o-', label='function', alpha=0.5, markerfacecolor='none')
integrator.hist.hist.plot_bar(label='hist')

ax.vlines(integrator.points.xedges.data(), -2, 2, linestyle='--', alpha=0.4, linewidth=0.5)
ax.set_ylim(-1.5, 1.5)

ax.legend(loc='lower right')

from mpl_tools.helpers import savefig
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '')
savefig(oname+'.png')

from gna.graphviz import savegraph
savegraph(fcn.sum, oname+'_graph.png')

plt.show()
