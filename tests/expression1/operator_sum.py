#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *
from gna.bundle import execute_bundles
from load import ROOT as R
R.GNAObject

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
args = parser.parse_args()

#
# Indices and their variants
# Each line has a syntax: 'short name', 'long name', 'index variants'
#
indices = [
    ('n', 'num',   ['1', '2']),
    ('a', 'alph',  ['a', 'b']),
    ('z', 'zyx',   ['X', 'Y']),
    ('b', 'bkg',   ['b1', 'b2'])
    ]

#
# Lib is a dictionary, that tells how the intermediate expressions should be named
#
#
lib = dict(
    sum = dict(expr='ta+tb+tc'),
    prod = dict(expr='ta*tb*tc'),
)

# The expression is a forumla
expr = ['ta()+tb[n]()+tc()', 'ta()*tb()*tc()']
# In order to parse a formula the Expression needs to know how to iterate over indices
a = Expression(expr, indices=indices)

# Print the expression formula before and after preprocessing
print(a.expressions_raw)
print(a.expressions)

# Parse the formula. The following line will execute the expression as python code and create all the objects, needed for its functioning.
a.parse()
# Assign the names for all the intermediate products, sums, etc.
a.guessname(lib, save=True)
# Print the tree to the terminal
a.tree.dump(True)

print()
# Provide the configuration. The configuration tells the Expression on how to build each variable/output.
cfg = NestedDict(
        ta = NestedDict(
            bundle = 'dummy',
            name = 'ta',
            input = False,
            size = 10,
            debug = False
            ),
        tb = NestedDict(
            bundle = 'dummy',
            name = 'tb',
            input = False,
            size = 10,
            debug = False
            ),
        tc = NestedDict(
            bundle = 'dummy',
            name = 'tc',
            input = False,
            size = 10,
            debug = False
            ),
        )
# Build the expression for given configuration. The output is a context, that contains the inputs and outputs.
context = ExpressionContext( cfg )
a.build(context)

from gna.bindings import OutputDescriptor
# Print inputs
print('Inputs')
print(context.inputs)

# Print outputs
print()
print('Outputs')
print(context.outputs)

if args.dot:
    # try:
    from gna.graphviz import GNADot

    graph = GNADot( context.outputs.ta )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
