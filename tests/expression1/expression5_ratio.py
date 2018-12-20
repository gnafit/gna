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

indices = [
    ('i', 'index',   ['1', '2', '3']),
    ('j', 'jndex',   ['A', 'B', 'C'])
    ]

lib = dict(
        ratio = dict(expr='top/bottom'),
)

expr = 'top[i]()/bottom[j]()'
cfg = NestedDict(
        top = NestedDict(
            bundle = 'arange',
            name = 'top',
            provides = ['top'],
            args = ( 1.0, 10, 1 )
            ),
        bottom = NestedDict(
            bundle = 'arange',
            name = 'bottom',
            provides = ['bottom'],
            args = ( 2.0, 20, 2 )
            ),
        )

a = Expression(expr, indices=indices)

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
context = ExpressionContext( cfg )
a.build(context)

from gna.bindings import OutputDescriptor
print( 'outputs:' )
print( context.outputs )

print( 'Ratio (should be 0.5):', context.outputs['ratio.1.A'].data() )

if args.dot:
    # try:
    from gna.graphviz import GNADot

    graph = GNADot( context.outputs.ratio['1.A'] )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
