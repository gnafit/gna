#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the 1d integrator (Gauss-Legendre) transformation

The script creates a Gaussian and then integrates it for each bin.
Usage:
    1. Simply run and plot:
        tests/elementary/test_integral_gl1d.py

    2. Create a graph:
        tests/elementary/test_integral_gl1d.py --dot output/integrator.dot
    and visualize it:
        xdot output/integrator.dot
"""

from __future__ import print_function
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_bar
import numpy as N
from load import ROOT as R
from constructors import Points
from gna.env import env
from argparse import ArgumentParser
from gna.parameters.printer import print_parameters
from mpl_tools import bindings

"""Parse arguments"""
parser = ArgumentParser()
parser.add_argument( '-o', '--order', type=int, default=3, help='integration order' )
parser.add_argument( '-g', '--gauss', type=float, nargs=2, default=[6.0, 1.0], help='gaussian mean and sigma' )
parser.add_argument( '-m', '--mean', type=float, help='gaussian mean' )
parser.add_argument( '-b', '--bins', type=float, nargs=3, default=[ 0.0, 12.001, 0.05 ], help='Bins: arange arguments (min, max, step)' )
parser.add_argument( '-l', '--legend', default='upper right', help='legend location' )
parser.add_argument( '--dot', help='write graphviz output' )
opts = parser.parse_args()

mean, sigma = opts.gauss

"""Initialize the function and environment"""
ns = env.globalns
ns.defparameter( 'BackgroundRate', central=0.0,   sigma=0.1 )
ns.defparameter( 'Mu',             central=1.0,   sigma=0.1 )
ns.defparameter( 'E0',             central=mean,  sigma=0.1 )
ns.defparameter( 'Width',          central=sigma, sigma=0.1 )
print_parameters( ns )

# Initialize the function of interest
fcn = R.GaussianPeakWithBackground()
fcn.rate.setLabel('Gaussian\n(function to integrate)')
# keep its output as variable
output=fcn.rate.rate

"""Initialize the integrator"""
# create array with bin edges
edges = N.arange(*opts.bins, dtype='d')

# create 1d integrator (sample points) for given edges and integration order
gl_int = R.GaussLegendre( edges, opts.order, edges.size-1 )
gl_int.points.setLabel('Integrator inputs')
gl_int.points.x.setLabel('E (points)')
gl_int.points.xedges.setLabel('E (bin edges)')
edges = gl_int.points.xedges.data()
widths = edges[1:]-edges[:-1]

# create the 1d integrator (histogram) for given sample points
gl_hist = R.GaussLegendreHist( gl_int )
gl_hist.hist.setLabel('Integrator (histogram)')

# knots = gl_int.points.x.data()
# y = N.exp(-(knots-mean)**2/2.0/sigma**2)/N.sqrt(2.0*N.pi)/sigma
# points = Points( y )
# output = points.points.points

"""Make fake gaussian data"""
# pass sample points as input to the function 'energy'
fcn.rate.E(gl_int.points.x)
# pass the function output to the histogram builder (integrator)
gl_hist.hist.f( output )
# read the histogram contents
hist_output = gl_hist.hist.hist
hist_output.setLabel('output histogram')
hist = hist_output.data()

"""Plot data"""
# init figure
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'f(x)' )
ax.set_title( 'Integrate Gaussian (%g, %g) with order %i'%( mean, sigma, opts.order ) )

# plot function versus sample points
ax.plot( gl_int.points.x.data(), fcn.rate.rate.data(), label='function' )
# plot histogram using GNA to matplotlib interface
hist_output.plot_bar(label='histogram (sum=%g)'%hist.sum(), alpha=0.5 )
# plot histogram manually
plot_bar( edges, hist/widths, label='histogram/binwidth', alpha=0.5 )

# add legend
ax.legend( loc=opts.legend)

# Our function of interest is a guassian and should give 1 when integrated
# Test it by summing the histogram bins
diff = hist.sum()-1
print( 'Integral-1:', diff )
print( N.fabs(diff)<1.e-8 and '\033[32mIntegration is OK!' or '\033[31mIntegration FAILED!', '\033[0m' )

if opts.dot:
    try:
        from gna.graphviz import GNADot
        graph = GNADot(gl_hist.hist)
        graph.write(opts.dot)
        print( 'Write output to:', opts.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

P.show()
