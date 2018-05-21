#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundle
from load import ROOT as R
from argparse import ArgumentParser
from gna.env import env
R.GNAObject

parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
args = parser.parse_args()

indices = [
    ('n', 'num',   ['1', '2']),
    ('a', 'alph',  ['a', 'b']),
    ('z', 'zyx',   ['X', 'Y', 'Z']),
    ('b', 'bkg',   ['b1', 'b2'])
    ]

lib = dict(
        wspec    = dict( expr = 'w2*spec' ),
        w2       = dict( expr='weight1*weight2' ),
        totalsum = dict( expr= 'sum:z' ),
        totalprod = dict( expr= 'prod:a' ),
)

expr = 'prod[a]| sum[z]| weight1[z]*weight2[a] * spec| enu()'
a = Expression(expr, indices=indices)

print(a.expression_raw)
print(a.expression)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        weight1 = NestedDict(
            bundle = 'dummyvar',
            variables = uncertaindict([
                ('weight1', (2, 0.1)),
                ],
                mode='percent'
                )
            ),
        weight2 = NestedDict(
            bundle = 'dummyvar',
            variables = uncertaindict([
                ('weight2', (3, 0.1)),
                ],
                mode='percent'
                )
            ),
        enu = NestedDict(
            bundle = 'dummy',
            name = 'enu',
            input = False,
            size = 10,
            debug = False
            ),
        spec = NestedDict(
            bundle = 'dummy',
            name = 'spec',
            input = True,
            size = 10,
            debug = False
            ),
        )
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters()
print( 'outputs:' )
print( context.outputs )

if args.dot:
    # try:
    from gna.graphviz import GNADot

    graph = GNADot( context.outputs.totalprod )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
