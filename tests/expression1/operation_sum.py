#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.expression_v01 import Expression_v01 as Expression, ExpressionContext_v01 as ExpressionContext
from gna.bundle import execute_bundles
from load import ROOT as R
from gna.configurator import NestedDict, uncertaindict
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
)

# The expression is a forumla
expr = [
        'tmp1=vara[a,n]*ta[a,n]()',
        'tmp2=sum[n]| varb[a] * tmp1',
        'res=tb[n]()*tmp1/tmp2'
        ]
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
            bundle = dict(name='dummy', version='v01'),
            name = 'ta',
            input = False,
            size = 10,
            debug = False
            ),
        tb = NestedDict(
            bundle = dict(name='dummy', version='v01'),
            name = 'tb',
            input = False,
            size = 10,
            debug = False
            ),
        va = NestedDict(
            bundle = dict(name='dummyvar', version='v01'),
            variables = uncertaindict([('vara', (2, 0.1))], mode='percent'),
            ),
        vb = NestedDict(
            bundle = dict(name='dummyvar', version='v01'),
            variables = uncertaindict([('varb', (3, 0.1))], mode='percent'),
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
    from gna.graphviz import GNADot, savegraph

    savegraph(next(context.outputs.ta.values(nested=True)), args.dot, rankdir='LR')
