#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.env import env
import constructors as C
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

dbg = R.DebugTransformation('Execute transformation function', 'Execute types function', 5.0)

print('Connect to input')
dbg.add_input(fcn.sum.sum)

print('Connect output')
integrator.add_input(dbg.transformations.back().outputs.back())

print('Read data 1st')
integrator.hist.hist.data()

print('Read data 2d')
integrator.hist.hist.data()

print('Done')

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

from gna.graphviz import savegraph
from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '')
savegraph(fcn.sum, oname+'_graph.png')

plt.show()
