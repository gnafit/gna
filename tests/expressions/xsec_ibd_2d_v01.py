#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Computes the integral:
    \int_{-1}^{+1} d\cos\theta \int_{E_i}^{E_j} dE \sigma(E_\nu(E, \cos\theta)) dE_\nu(E, \cos \theta)/dE
    for each energy bin.
"""

# Usage:
#   - Just run and show the figure:
#      tests/bundle/xsec_ibd_2d_v01.py -s
#
#   - Save also the graphviz histogram (requires pygraphviz python module)
#      tests/bundle/xsec_ibd_2d_v01.py -s --dot output/xsec.dot
#   - Open dot file (needs xdot):
#      xdot output/xsec.dot
#   - Or just plot it (needs dot from graphviz):
#      dot -O -Tpdf -v output/xsec.dot

from __future__ import print_function
#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
args = parser.parse_args()

#
# Import libraries
#
from gna.expression import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundle
from load import ROOT as R
from gna.env import env
from matplotlib import pyplot as P
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

#
# Expressions to evaluate. May be a string or a list of string. The expressions are evaluated in the common context.
#
# Here are the ground rules:
#   - 'somename' is a variable, or just a double number
#   - 'somename()' is a transformation output, holding and array.
#   - 'somename( other() )':
#     the function call with arguments means that the output of arguments is connected to the input of a function.
#     in this case the output 'other()' is connected as input to 'somename()'
#
# '|' is a short cut for a function call:
#   - 'a| b' is 'a(b)'
#   - 'c(a| b)' is the same as 'c(a(b))'. I.e. balanced parentheses are respected.
#
# The actual variables/transformations are provided by the bundles (see the configuration below). The
# expression only determines how to connect them.
expr =[
        # - evis() is visible energy, which is provided by the integrator. These are all the points needed
        # to compute the integrals for each bin.
        #
        # - ee() is positron energy provided by cross section, which is computed from evis().
        #
        # - enu() is neutrino energy provided by cross section, which is computed from ee().
        #
        # in first rexpression we feed 'evis()' to 'ee() and then to enu()'
        'enu| ee(evis()), ctheta()',
        # connect jacobian
        'jacobian(enu(), ee(), ctheta())',
        # the ibd_xsec() is a cross section is provided by the cross section bundle. It depends on 'enu()' and 'costheta()'.
        'kinint2| ibd_xsec(enu(), ctheta()) * jacobian()'
        # 'kinint2' is 2d integration function provided by the integrator. It is needed to convert
        # the cross section, computed in each point to a histogram. The integration is done for each evis bin and for ctheta=[-1,1]
        ]

# Initialize the expression and indices
a = Expression(expr, indices)

# Dump the information
print(a.expressions_raw)
print(a.expressions)

# Parse the expression
a.parse()
# The next step is needed to name all the intermediate variables.
a.guessname(lib, save=True)
# Dump the tree.
a.tree.dump(True)

#
# At this point what you have is a dependency tree with variables, transformations (all indexed),
# but without actual implementation. We add the implementation on a next step.
#

print()
# Here is the configuration
cfg = NestedDict(
        # Configuration of the 'kinint' variable. It says
        kinint2 = NestedDict(
            # that one need to execute the bundle 'integral_1d_v01'
            # that will create and provide the necessary transformations
            bundle   = 'integral_2d1d_v01',
            # The following lines are the bundle options:
            # - the integration variable name
            variables = ('evis', 'ctheta'),
            # - the bin edges
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            # - the integration order for each X bin (or fo all of the bins) (Gauss-Legendre)
            xorders   = 3,
            # - the integration order for all Y bins (Gauss-Legendre)
            yorder   = 5,
            # - this line says that the bundle will create 'evis' and 'ctheta' output in addition to 'kinint2'
            provides = [ 'evis', 'ctheta' ]
            ),
        # This bundle configuration will be triggered in order to build 'ibd_xsec()' output.
        ibd_xsec = NestedDict(
            # bundle name to be executed
            bundle = 'xsec_ibd_v01',
            # and its parameters:
            # - the IBD cross section order (0 for zero-th or 1 the first). First is not yet implemented.
            order = 1,
            # this line says that the bundle will provide the 'ee' - positron energy as additional output.
            provides = [ 'ibd_xsec', 'ee', 'enu', 'jacobian' ]
            )
        )
#
# Put the expression into context
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

# Print the list of outputs, stored for the expression
from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

#
# Do some plots
#
# Initialize figure
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'Visible energy, MeV' )
ax.set_ylabel( r'$\sigma$' )
ax.set_title( 'IBD cross section (1st order)' )

# Plot
context.outputs.kinint2.plot_hist( label='Integrated cross section' )

ax.legend(loc='upper left')

# #
# # Check, that unraveled Enu is always gowing
# #
# enu  = context.outputs.enu.data()

# fig = P.figure()
# ax = P.subplot( 111 )
# ax.minorticks_on()
# ax.grid()
# ax.set_xlabel(L.u('enu'))
# ax.set_ylabel(L('{enu} step'))
# ax.set_title(L('Check {enu} step'))

# idx = N.arange(enu.shape[0])
# for i, e in enumerate(enu.T):
    # ax.plot(idx, e, '-', label='Slice %i'%i)

# ax.legend(loc='upper left')

if args.show:
    P.show()

import IPython
IPython.embed()

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
