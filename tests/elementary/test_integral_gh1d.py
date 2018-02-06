#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the SelfPower transformation"""

from __future__ import print_function
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_bar
import numpy as N
from load import ROOT as R
from constructors import Points
from gna.env import env
from argparse import ArgumentParser
from gna.parameters.printer import print_parameters

"""Parse arguments"""
parser = ArgumentParser()
parser.add_argument( '-o', '--order', type=int, default=3, help='integration order' )
parser.add_argument( '-g', '--gauss', type=float, nargs=2, default=[6.0, 1.0], help='gaussian mean and sigma' )
parser.add_argument( '-m', '--mean', type=float, help='gaussian mean' )
parser.add_argument( '-b', '--bins', type=float, nargs=3, default=[ 0.0, 12.001, 0.05 ], help='Bins: arange arguments (min, max, step)' )
parser.add_argument( '-l', '--legend', default='upper right', help='legend location' )
opts = parser.parse_args()

mean, sigma = opts.gauss

"""Initialize the function and environment"""
ns = env.globalns
ns.defparameter( 'BackgroundRate', central=0.0,   sigma=0.1 )
ns.defparameter( 'Mu',             central=1.0,   sigma=0.1 )
ns.defparameter( 'E0',             central=mean,  sigma=0.1 )
ns.defparameter( 'Width',          central=sigma, sigma=0.1 )
print_parameters( ns )

fcn = R.GaussianPeakWithBackground()
output=fcn.rate.rate

"""Initialize the integrator"""
edges = N.arange(*opts.bins, dtype='d')

gl_int = R.GaussLegendre( edges, opts.order, edges.size-1 )
edges = gl_int.points.xedges.data()
widths = edges[1:]-edges[:-1]

gl_hist = R.GaussLegendreHist( gl_int )

# knots = gl_int.points.x.data()
# y = N.exp(-(knots-mean)**2/2.0/sigma**2)/N.sqrt(2.0*N.pi)/sigma
# points = Points( y )
# output = points.points.points

"""Make fake gaussian data"""
fcn.rate.E(gl_int.points.x)
gl_hist.hist.f( output )
hist = gl_hist.hist.hist.data()

"""Plot data"""
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'f(x)' )
ax.set_title( 'Integrate Gaussian (%g, %g) with order %i'%( mean, sigma, opts.order ) )

ax.plot( gl_int.points.x.data(), fcn.rate.rate.data(), label='function' )
plot_bar( edges, hist, label='histogram (sum=%g)'%hist.sum(), alpha=0.5 )
plot_bar( edges, hist/widths, label='histogram/binwidth', alpha=0.5 )

ax.legend( loc=opts.legend)

diff = hist.sum()-1
print( 'Integral-1:', diff )
print( N.fabs(diff)<1.e-8 and '\033[32mIntegration is OK!' or '\033[31mIntegration FAILED!', '\033[0m' )

P.show()
