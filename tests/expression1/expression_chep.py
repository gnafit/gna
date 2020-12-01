#!/usr/bin/env python

from gna.expression import *
from gna.bundle import execute_bundles
from load import ROOT as R
from gna.configurator import uncertaindict
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
        ('i', 'index_i', ('i1', 'i2')),
        ('j', 'index_j', ('j1',)),
        ('k', 'index_k', ('k1', 'k2')),
    ]

#
# Lib is a dictionary, that tells how the intermediate expressions should be named
#
#
lib = dict(
        scaled_vec = dict(
            expr  = 'scale*vec',
            label = 'scaled vec\ni={index_i}, j={index_j}\n({weight_label})'
            ),
        sum = dict(
            expr  = 'offset+scaled_vec',
            label = 'scaled_vec+offset\ni={index_i}, j={index_j}'
            ),
        result  = dict(
            expr = 'sum:k|function',
            label = 'sum function over k\ni={index_i}, j={index_j}'
            )
)

# The expression is a forumla
expr = 'sum[k]| function[k]| scale[i] * vec[j]() + offset()'
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
        vec = NestedDict(
            bundle = 'dummy',
            name = 'vec',
            label = 'array\nj={index_j}',
            input = False,
            size = 10,
            debug = False
            ),
        offset = NestedDict(
            bundle = 'dummy',
            name = 'offset',
            input = False,
            size = 10,
            debug = False
            ),
        function = NestedDict(
            bundle = 'dummy',
            name = 'function',
            label = 'function\ni={index_i}, j={index_j}, k={index_k}',
            input = 1,
            size = 10,
            debug = False
            ),
        scale = NestedDict(
            bundle = 'dummyvar',
            variables = uncertaindict([
                ('scale', (2, 0.1)),
                ],
                mode='percent'
                )
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

    graph = GNADot( context.outputs.offset )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
