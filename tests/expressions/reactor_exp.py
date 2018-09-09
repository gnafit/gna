#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
indices = [
    ('r', 'reactor',     ['DB', 'LA1', 'LA2']),
    ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']),
    ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
    ]

indices = [
    ('r', 'reactor',     ['DB']),
    ('d', 'detector',    ['AD11']),
    ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
    ]

#
# Intermediate options (empty for now)
#
lib = dict()

expr =[
        'enu| ee(evis()), ctheta()',
        'jacobian(enu(), ee(), ctheta())',
        # 'kinint2| ibd_xsec(enu(), ctheta()) * jacobian()'
        'anuspec(enu()) * oscprob[c,d,r]( enu() ) * ibd_xsec(enu(), ctheta())'
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
        kinint2 = NestedDict(
            bundle   = 'integral_2d1d_v01',
            variables = ('evis', 'ctheta'),
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            xorders   = 3,
            yorder   = 5,
            provides = [ 'evis', 'ctheta' ]
            ),
        ibd_xsec = NestedDict(
            bundle = 'xsec_ibd_v01',
            order = 1,
            provides = [ 'ibd_xsec', 'ee', 'enu', 'jacobian' ]
            ),
        oscprob = NestedDict(
            bundle = 'oscprob_v01',
            name = 'oscprob',
            input = True,
            size = 10,
            debug = False
            ),
        anuspec = NestedDict(
            bundle = 'reactor_anu_spectra_v02',
            name = 'anuspec',
            filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                            'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],

            strategy = dict( underflow='constant', overflow='extrapolate' ),
            edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
            ),
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

