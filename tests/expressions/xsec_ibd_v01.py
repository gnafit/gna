#!/usr/bin/env python

r"""Computes the integral:
    \theta \int_{E_i}^{E_j} dE \sigma_0(E_\nu(E))
    for each energy bin.
"""

# Usage:
#   - Just run and show the figure:
#      tests/bundle/xsec_ibd_v01.py -s
#
#   - Save also the graphviz histogram (requires pygraphviz python module)
#      tests/bundle/xsec_ibd_v01.py -s --dot output/xsec.dot
#   - Open dot file (needs xdot):
#      xdot output/xsec.dot
#   - Or just plot it (needs dot from graphviz):
#      dot -O -Tpdf -v output/xsec.dot

#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument('-e', '--energy_choice', choices=['enu', 'ee', 'evis'], default='evis', 
                    help='Choice against what energy xsec will be ploted')
parser.add_argument('-d', '--differential_xsec', action='store_true', 
                    help='Plot differential cross section')
args = parser.parse_args()
energy_use = args.energy_choice

#
# Import libraries
#
from gna.expression import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundles
from load import ROOT as R
from gna.env import env
from matplotlib import pyplot as P
import numpy as N
from mpl_tools import bindings
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
        'enu| ee| evis()',
        # the ibd_xsec() is a cross section is provided by the cross section bundle. It depends on 'ee()'.
        'kinint| ibd_xsec(ee())'
        # 'kinint' is an integration function provided by the integrator. It is needed to convert
        # the cross section, computed in each point to a histogram.
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
# Dump the trees
for i, tree in enumerate(a.trees):
    print('Tree', i)
    tree.dump(True)

#
# At this point what you have is a dependency tree with variables, transformations (all indexed),
# but without actual implementation. We add the implementation on a next step.
#

print()
# Here is the configuration
cfg = NestedDict(
        # Configuration of the 'kinint' variable. It says
        kinint = NestedDict(
            # that one need to execute the bundle 'integral_1d_v01'
            # that will create and provide the necessary transformations
            bundle   = 'integral_1d_v01',
            # The following lines are the bundle options:
            # - the integration variable name
            variable = 'evis',
            # - the bin edges
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            # - the integration order for each bin (or fo all of the bins) (Gauss-Legendre)
            orders   = 3,
            # - this line says that the bundle will create 'evis' output in addition to 'kinint'
            provides = [ 'evis' ]
            ),
        # This bundle configuration will be triggered in order to build 'ibd_xsec()' output.
        ibd_xsec = NestedDict(
            # bundle name to be executed
            bundle = 'xsec_ibd_v01',
            # and its parameters:
            # - the IBD cross section order (0 for zero-th or 1 the first). First is not yet implemented.
            order = 0,
            # this line says that the bundle will provide the 'ee' - positron energy as additional output.
            provides = [ 'ibd_xsec', 'ee', 'enu' ]
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
ax.set_title("IBD cross section vs {} in zero order".format(energy_use))

# Get evis and cross section
evis = context.outputs.evis.data()
enu = context.outputs.enu.data()
xsec = context.outputs.ibd_xsec.data()
integrated = context.outputs.kinint.data()

# Plot
if args.differential_xsec:
    if energy_use == 'evis':
        ax.plot( evis, xsec, label='Differential IBD cross section' )
    elif energy_use == 'enu':
        ax.plot(enu, xsec, label='Differential IBD cross section')


context.outputs.kinint.plot_hist( label='Integrated cross section' )

ax.legend(loc='upper left')

if args.show:
    P.show()

#
# Dump the histogram to a dot graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot(context.outputs.ee, joints=False)
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise
