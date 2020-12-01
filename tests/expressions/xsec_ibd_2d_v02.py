#!/usr/bin/env python

r"""Computes the integral:
    \int_{-1}^{+1} d\cos\theta \int_{E_i}^{E_j} dE \sigma(E_\nu(E, \cos\theta)) dE_\nu(E, \cos \theta)/dE
    for each energy bin.
"""

#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
args = parser.parse_args()

from gna.expression.expression_v01 import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundles
from load import ROOT as R
from gna.env import env
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as N
from mpl_tools import bindings
from gna.labelfmt import formatter as L
R.GNAObject

#
# Define the indices (empty for current example)
#
indices = []

#
# Intermediate options (empty for now)
#
lib = dict()

formulas =[
        # - evis() is visible energy, which is provided by the integrator. These are all the points needed
        # to compute the integrals for each bin.
        # - ee() is positron energy provided by cross section, which is computed from evis().
        # - enu() is neutrino energy provided by cross section, which is computed from ee().
        # in first rformulasession we feed 'evis()' to 'ee() and then to enu()'
        'enu| ee(evis()), ctheta()',
        'jacobian(enu(), ee(), ctheta())',
        'project| kinint2| ibd_xsec(enu(), ctheta()) * jacobian()',
        ]

expr = Expression_v01(formulas, indices)

# Dump the information
print(expr.expressions_raw)
print(expr.expressions)

# Parse the expression
expr.parse()
# The next step is needed to name all the intermediate variables.
expr.guessname(lib, save=True)
# Dump the tree.
expr.tree.dump(True)

print()
# Here is the configuration
cfg = NestedDict(
        kinint2 = NestedDict(
            bundle   = dict(name='integral_2d', version='v01'),
            variables = ('evis', 'ctheta'),
            xedges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            yedges    = N.linspace(-1.0, 1.0, 9, dtype='d'),
            xorders   = 3,
            yorders   = 3,
            instances = {
                'kinint2': '2d integral'
                }
            ),
        ibd_xsec = NestedDict(
            bundle = dict(name='xsec_ibd', version='v02'),
            order = 1,
            ),
        project = NestedDict(
            bundle = dict(name='simple', version='v01', major=''),
            name = 'project',
            actions = dict(
                object =lambda: R.SumAxis(0),
                input  =lambda obj: (obj.add_transformation(), obj.add_input())[1],
                output =lambda obj: obj.transformations.back().outputs[0],
                )
            ),
        )
#
# Initialize bundles
#
context = ExpressionContext_v01(cfg, ns=env.globalns)
expr.build(context)

# Print the list of outputs, stored for the expression
from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

#
# Do some plots
#
# Initialize figure
fig = plt.figure()
ax = plt.subplot(111, projection='3d')
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'Visible energy, MeV' )
ax.set_ylabel( r'$\sigma$' )
ax.set_title( 'IBD cross section (1st order)' )

# Plot
context.outputs.kinint2.plot_wireframe(colorbar=True, cmap=True)

ax.legend(loc='upper left')

if args.show:
    plt.show()

#
# Dump the histogram to a dot graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot(context.outputs.ee, joints=True)
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise
