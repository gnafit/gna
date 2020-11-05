#!/usr/bin/env python

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

from matplotlib import pyplot as P
from mpl_tools.helpers import plot_bar
import numpy as N
from load import ROOT as R
from gna.constructors import Points, Histogram, Histogram2d, stdvector
from gna.env import env
from argparse import ArgumentParser
from gna.parameters.printer import print_parameters
from mpl_tools import bindings
from mpl_toolkits.mplot3d import Axes3D

"""Parse arguments"""
parser = ArgumentParser()
parser.add_argument( '-o', '--orders', type=int, nargs=2, default=(5,6), help='integration order' )
parser.add_argument( '-x', '--xbins', type=float, nargs=3, default=[ 0.0, 7.001, 1.0 ], help='Bins: arange arguments (min, max, step)' )
parser.add_argument( '-y', '--ybins', type=float, nargs=3, default=[ 1.0, 8.001, 2.0 ], help='Bins: arange arguments (min, max, step)' )
parser.add_argument( '--ab', type=float, nargs=2, default=(1.0, 2.0), help='function parameters' )
parser.add_argument( '--input-edges', action='store_true', help='pass edges as input' )
parser.add_argument( '-M', '--mode', default='gl2', choices=['gl2', 'gl21'], help='integration mode' )
# parser.add_argument( '-l', '--legend', default='upper right', help='legend location' )
parser.add_argument( '-d', '--dump', action='store_true', help='dump integrator' )
parser.add_argument( '--dot', help='write graphviz output' )
opts = parser.parse_args()

"""Initialize the integrator"""
# create array with bin edges
xedges = N.arange(*opts.xbins, dtype='d')
yedges = N.arange(*opts.ybins, dtype='d')

# create 2d integrator (sample points) for given edges and integration order
integrators = dict(gl2=R.Integrator2GL, gl21=R.Integrator21GL)
Integrator = integrators[opts.mode]

mode21= opts.mode=='gl21'

if not mode21:
    if opts.input_edges:
        edges_in = Histogram2d(xedges, yedges, xedges[:-1,None]*yedges[:-1])
        integrator = Integrator(xedges.size-1, opts.orders[0], yedges.size-1, opts.orders[1])
        integrator.points.edges(edges_in)
    else:
        integrator = Integrator(xedges.size-1, opts.orders[0], xedges, yedges.size-1, opts.orders[1], yedges)
else:
    if yedges.size!=2:
        raise Exception('In GL21 mode there should be only one bin over second axis')
    if opts.input_edges:
        xedges_in = Histogram(xedges, xedges[:-1])
        integrator = Integrator(xedges.size-1, opts.orders[0], None, opts.orders[1], yedges[0], yedges[1])
        integrator.points.xedges(xedges_in)
    else:
        integrator = Integrator(xedges.size-1, opts.orders[0], xedges, opts.orders[1], yedges[0], yedges[1])

integrator.points.setLabel('Integrator inputs')
integrator.points.y.setLabel('X (points)')
integrator.points.xedges.setLabel('X (edges)')
integrator.points.y.setLabel('Y (points)')
if not mode21:
    integrator.points.yedges.setLabel('Y (edges)')
integrator.hist.setLabel('Integrator (histogram)')

xedges = integrator.points.xedges.data()
xwidths = xedges[1:]-xedges[:-1]

xmesh = integrator.points.xmesh.data()
ymesh = integrator.points.ymesh.data()

"""Make function"""
a, b = opts.ab
def fcn(x, y):
    return N.sin(a*x+b*y)

def fcn_int(x1, x2, y1, y2):
    weight = -1.0/(a*b)

    s1 = N.sin( a*x2 + b*y2 )
    s2 = N.sin( a*x1 + b*y1 )
    s3 = N.sin( a*x2 + b*y1 )
    s4 = N.sin( a*x1 + b*y2 )

    return weight*(s1+s2-s3-s4)

# def fcn(x, y):
    # return a*x+b*y

# def fcn_int(x1, x2, y1, y2):
    # return 0.5*(y2-y1)*(x2-x1)*(a*(x2+x1)+b*(y2+y1))

# def fcn(x, y):
    # return a + x*0.0

# def fcn_int(x1, x2, y1, y2):
    # return a*(x2-x1)*(y2-y1)

def integr(x, y):
    x, y = N.meshgrid(x, y, indexing='ij')
    x1 = x[:-1,:-1]
    x2 = x[1: ,1:]
    y1 = y[:-1,:-1]
    y2 = y[1:,1: ]

    return fcn_int(x1, x2, y1, y2)

fcn_values = fcn(xmesh, ymesh)
integrals  = integr(xedges, yedges)

fcn_o = Points(fcn_values)
fcn_output=fcn_o.single()
integrator.hist.f(fcn_output)
hist_output = integrator.hist.hist
hist_output.setLabel('output histogram')
hist = hist_output.data()

"""Self test of integration"""
from scipy.integrate import dblquad
ix, iy = 4, min(yedges.size-2, 2)
x1, x2 = xedges[ix:ix+2]
y1, y2 = yedges[iy:iy+2]

int_s  = dblquad(lambda y, x: fcn(x, y), x1, x2, y1, y2)[0]
int_a1 = integr( [x1, x2], [y1, y2] )[0,0]
int_a2 = integrals[ix, iy]

print('Integration self check')
print( 'a, b', a, b )
print( 'x', x1, x2 )
print( 'y', y1, y2 )
print( 'Scipy:', int_s)
print( 'Analytic:', int_a1, int_a2 )
print( 'Diff (scipy-analytic):', int_s-int_a1 )
print( 'Diff (analytic):', int_a1-int_a2 )

print()
print('Integration check')
print('Analytic integrals')
print( integrals )
print('Numeric integrals')
print( hist )

if mode21:
    OK = N.allclose(integrals.T[0], hist)
else:
    OK = N.allclose(integrals, hist)
print(OK and '\033[32mIntegration is OK\033[0m' or '\033[31mIntegration failed\033[0m')

if opts.dump:
    integrator.dump()
    # print('Abscissas:', integrator.points.x.data())
    # print('Widths:', widths)

"""Plot data"""
fig = P.figure()
ax = P.subplot( 111, projection='3d' )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( '' )
ax.plot_wireframe(xmesh, ymesh, fcn_values)

# # init figure
# fig = P.figure()
# ax = P.subplot( 111 )
# ax.minorticks_on()
# ax.grid()
# ax.set_xlabel( 'x' )
# ax.set_ylabel( 'f(x)' )
# ax.set_title( 'Integrate exponential (%s)'%(opts.mode) )

# baropts = dict(alpha=0.5)
# # plot function versus sample points
# ax.plot( integrator.points.x.data(), fcn_output.data(), '-', label='function' )
# # plot histogram using GNA to matplotlib interface
# hist_output.plot_bar(label='histogram (sum=%g)'%hist.sum(), **baropts)
# # plot histogram manually
# plot_bar( edges, hist/widths, label='histogram/binwidth', **baropts)

# add legend
# ax.legend(loc=opts.legend)

# # Our function of interest is a guassian and should give 1 when integrated
# # Test it by summing the histogram bins
# diff = hist.sum()-integral
# print('Integral (analytic)', integral)
# print('Integral (analytic, sum)', integrals.sum())
# print('Diff (Integral - %g):'%integral, diff)
# # print('Integrals (analytic)', integrals)
# # print('Integrals (calc)', hist)
# adiff = N.fabs(integrals-hist).sum()
# print('Diffs (abssum):', adiff)
# print( N.fabs(diff)<1.e-8 and adiff<1.e-8 and '\033[32mIntegration is OK!' or '\033[31mIntegration FAILED!', '\033[0m' )

if opts.dot:
    try:
        from gna.graphviz import GNADot
        graph = GNADot(integrator.hist)
        graph.write(opts.dot)
        print( 'Write output to:', opts.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

P.show()
