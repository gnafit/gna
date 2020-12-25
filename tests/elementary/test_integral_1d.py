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
from gna.constructors import Points, Histogram, stdvector
from gna.env import env
from argparse import ArgumentParser
from gna.parameters.printer import print_parameters
from mpl_tools import bindings

"""Parse arguments"""
parser = ArgumentParser()
parser.add_argument( '-o', '--order', type=int, default=3, help='integration order' )
parser.add_argument( '-b', '--bins', type=float, nargs=3, default=[ 0.0, 12.001, 0.05 ], help='Bins: arange arguments (min, max, step)' )
parser.add_argument( '-l', '--legend', default='upper right', help='legend location' )
parser.add_argument( '--input-edges', action='store_true', help='pass edges as input' )
parser.add_argument( '-M', '--mode', default='gl', choices=['gl', 'rect_left', 'rect', 'rect_right', 'trap', 'gl21'], help='integration mode' )
parser.add_argument( '-d', '--dump', action='store_true', help='dump integrator' )
parser.add_argument( '--dot', help='write graphviz output' )
opts = parser.parse_args()


"""Initialize the function and environment"""
ns = env.globalns
ns.defparameter( 'alpha', central=-0.5, sigma=0.1 )
print_parameters( ns )

names = stdvector(('alpha',))
fcn_we = R.WeightedSum(names, names)
fcn_input=fcn_we.sum.alpha

fcn_e = R.Exp()
fcn_e.exp.points(fcn_we.sum.sum)
fcn_output = fcn_e.exp.result

"""Initialize the integrator"""
# create array with bin edges
edges = N.arange(*opts.bins, dtype='d')

alpha=ns['alpha'].value()
a, b = edges[0], edges[-1]
integral = (N.exp(alpha*b) - N.exp(alpha*a))/alpha
aa, bb = edges[:-1], edges[1:]
integrals = (N.exp(alpha*bb) - N.exp(alpha*aa))/alpha

#21d mode
mode21 = opts.mode=='gl21'

# create 1d integrator (sample points) for given edges and integration order
integrators = dict(gl=R.IntegratorGL, rect=R.IntegratorRect, trap=R.IntegratorTrap, gl21=R.Integrator21GL)
Integrator = integrators[ opts.mode.split('_', 1)[0] ]
if '_' in opts.mode:
    iopts = opts.mode.rsplit('_', 1)[-1],
elif mode21:
    iopts = (3, 0.0, 1.0)
else:
    iopts = tuple()

if opts.input_edges:
    edges_in = Histogram(edges, edges[:-1])
    integrator = Integrator(edges.size-1, opts.order, R.nullptr, *iopts)
    integrator.points.edges(edges_in)
else:
    integrator = Integrator(edges.size-1, opts.order, edges, *iopts)

integrator.points.setLabel('Integrator inputs')
# integrator.points.x.setLabel('E (points)')
# integrator.points.xedges.setLabel('E (bin edges)')
if mode21:
    integrator.points.y.setLabel('Y')
integrator.hist.setLabel('Integrator (histogram)')

edges = integrator.points.xedges.data()
widths = edges[1:]-edges[:-1]

"""Make fake gaussian data"""
# pass sample points as input to the function 'energy'
if mode21:
   integrator.points.xmesh >> fcn_input
else:
   integrator.points.x >> fcn_input
# pass the function output to the histogram builder (integrator)
fcn_output >> integrator.hist.f

# read the histogram contents
hist_output = integrator.hist.hist
# hist_output.setLabel('output histogram')
hist = hist_output.data()

if opts.dump:
    integrator.dump()
    print('Abscissas:', integrator.points.x.data())
    print('Widths:', widths)
    print('Centers:', integrator.points.xcenters.data())

"""Plot data"""
# init figure
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'f(x)' )
ax.set_title( 'Integrate exponential (%s)'%(opts.mode) )

baropts = dict(alpha=0.5)
# plot function versus sample points
ax.plot( integrator.points.x.data(), fcn_output.data(), '-', label='function' )
# plot histogram using GNA to matplotlib interface
hist_output.plot_bar(label='histogram (sum=%g)'%hist.sum(), **baropts)
# plot histogram manually
plot_bar( edges, hist/widths, label='histogram/binwidth', **baropts)

# add legend
ax.legend(loc=opts.legend)

# Our function of interest is a guassian and should give 1 when integrated
# Test it by summing the histogram bins
diff = hist.sum()-integral
print('Integral (analytic)', integral)
print('Integral (analytic, sum)', integrals.sum())
print('Diff (Integral - %g):'%integral, diff)
# print('Integrals (analytic)', integrals)
# print('Integrals (calc)', hist)
adiff = N.fabs(integrals-hist).sum()
print('Diffs (abssum):', adiff)
print( N.fabs(diff)<1.e-8 and adiff<1.e-8 and '\033[32mIntegration is OK!' or '\033[31mIntegration FAILED!', '\033[0m' )

if opts.dot:
    try:
        from gna.graphviz import savegraph
        savegraph(integrator.hist, opts.dot)
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

P.show()
