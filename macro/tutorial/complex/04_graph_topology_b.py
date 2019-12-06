#!/usr/bin/env python

from __future__ import print_function
from tutorial import tutorial_image_name, savegraph
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
p1a = ns.defparameter('fcn1.a', central=1.0, free=True, label='weight 1')
p1b = ns.defparameter('fcn1.b', central=0.2, free=True, label='weight 2')
p1k = ns.defparameter('fcn1.k', central=8.0, free=True, label='argument scale')

p2a = ns.defparameter('fcn2.a', central=0.5, free=True, label='weight 1')
p2b = ns.defparameter('fcn2.b', central=0.1, free=True, label='weight 2')
p2k = ns.defparameter('fcn2.k', central=16.0, free=True, label='argument scale')

# Print the list of parameters
ns.printparameters(labels=True)
print()

# Define binning and integration orders
nbins = 20
x_edges = np.linspace(-np.pi, np.pi, nbins+1, dtype='d')
x_widths = x_edges[1:]-x_edges[:-1]
x_width_ratio = x_widths/x_widths.min()
orders = 4

# Initialize histogram
hist = C.Histogram(x_edges)

# Initialize integrator
integrator = R.IntegratorGL(nbins, orders)
integrator.points.edges(hist.hist.hist)
int_points = integrator.points.x

# Initialize sine and two cosines
sin_t = R.Sin(int_points)
cos1_arg = C.WeightedSum( ['fcn1.k'], [int_points] )
cos2_arg = C.WeightedSum( ['fcn2.k'], [int_points] )

cos1_t = R.Cos(cos1_arg.sum.sum)
cos2_t = R.Cos(cos2_arg.sum.sum)

# Initalizes both weighted sums
fcn1   = C.WeightedSum(['fcn1.a', 'fcn1.b'], [sin_t.sin.result, cos1_t.cos.result])
fcn2   = C.WeightedSum(['fcn2.a', 'fcn2.b'], [sin_t.sin.result, cos2_t.cos.result])

# Create two debug transformations to check the execution
dbg1a = R.DebugTransformation('before int1', 1.0)
dbg2a = R.DebugTransformation('before int2', 1.0)
dbg1b = R.DebugTransformation('after int1', 1.0)
dbg2b = R.DebugTransformation('after int2', 1.0)

# Conect inputs (debug 1)
print('Connect to inputs')
dbg1a.add_input(fcn1.sum.sum)
dbg2a.add_input(fcn2.sum.sum)
print()

# Conect outputs (debug 1) to the integrators
print('Connect outputs')
int1 = integrator.transformations.back()
integrator.add_input(dbg1a.debug.target)
integrator.add_input(dbg2a.debug.target)
print()

# Connect inputs of debug transformations
print('Connect to inputs')
dbg1b.add_input(int1.hist)
dbg2b.add_input(int1.hist_02)
print()

# Read outputs of debug transformations
print('Read data 1: 1st')
dbg1b.debug.target.data()
print()

print('Read data 1: 2d')
dbg1b.debug.target.data()
print('    <no functions executed>')
print()

print('Read data 2d 1st')
dbg2b.debug.target.data()
print()

print('Read data 2d 2d')
dbg2b.debug.target.data()
print('    <no functions executed>')
print()

# Change paramter k1 which affects only the first integrand
print('Change k1')
p1k.set(9)

# Read outputs of debug transformations
print('Read data 1: 1st')
dbg1b.debug.target.data()
print()

print('Read data 1: 2d')
dbg1b.debug.target.data()
print('    <no functions executed>')
print()

print('Read data 2d 1st')
dbg2b.debug.target.data()
print()

print('Read data 2d 2d')
dbg2b.debug.target.data()
print('    <no functions executed>')
print()

print('Done')

integrator.print()
print()

print(dbg1b.debug.target.data().sum(), dbg2b.debug.target.data().sum())

# # Label transformations
hist.hist.setLabel('Input histogram\n(bins definition)')
integrator.points.setLabel('Sampler\n(Gauss-Legendre)')
sin_t.sin.setLabel('sin(x)')
cos1_arg.sum.setLabel('k1*x')
cos2_arg.sum.setLabel('k2*x')
cos1_t.cos.setLabel('cos(k1*x)')
cos2_t.cos.setLabel('cos(k2*x)')
int1.setLabel('Integrator\n(convolution)')
fcn1.sum.setLabel('a1*sin(x) + b1*cos(k1*x)')
fcn2.sum.setLabel('a2*sin(x) + b2*cos(k2*x)')
dbg1a.debug.setLabel('Debug 1\n(before integral 1)')
dbg2a.debug.setLabel('Debug 2\n(before integral 2)')
dbg1b.debug.setLabel('Debug 1\n(after integral 1)')
dbg2b.debug.setLabel('Debug 2\n(after integral 2)')




savegraph(int_points, tutorial_image_name('png', suffix='graph'), rankdir='TB')

plt.show()
