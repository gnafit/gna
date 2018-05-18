#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *
from gna.bundle import execute_bundle
from load import ROOT as R
R.GNAObject

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
args = parser.parse_args()

indices = [
    ('n', 'num',   ['1', '2', '3', '4', '5', '6', '7', '8'])
    ]

lib = dict(
        finalsum = dict(expr='sum:n'),
)

expr = 'input[n]( enu() )'
cfg = NestedDict(
        enu = NestedDict(
            bundle = 'dummy',
            name = 'enu',
            input = False,
            size = 10,
            debug = False
            ),
        input = NestedDict(
            bundle = 'dummy',
            name = 'input',
            input = True,
            size = 10,
            debug = False
            ),
        )
for i in range(6):
    fcn = 'fcn%i'%i
    expr = '%s| %s'%(fcn, expr)
    cfg[fcn] = NestedDict(
            bundle = 'dummy',
            name =fcn,
            input = True,
            size = 10,
            debug = False
            )
expr = 'sum[n]| '+expr

a = Expression(expr, indices=indices)

print(a.expression_raw)
print(a.expression)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
context = ExpressionContext( cfg )
a.build(context)

from gna.bindings import OutputDescriptor
print( 'outputs:' )
print( context.outputs )

if args.dot:
    # try:
    from gna.graphviz import GNADot

    graph = GNADot( context.outputs.finalsum )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
