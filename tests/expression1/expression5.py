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
    ( 'r', 'reactor', ('DB', 'LA1', 'LA2') ),
    ( 'd', 'detector', ('AD11', 'AD12', 'AD21') ),
    ( 'c', 'component', ('comp0', 'comp1', 'comp2') )
    ]

def rules( nidx ):
    if nidx.indices['c'].current == 'comp0':
        nidx.indices['r'].current = 'all'
        nidx.indices['d'].current = 'all'
    return nidx

lib = dict(
    sums = dict( expr = 'sum:n' )
)

expr = 'enu()'
a = Expression(expr, indices=indices)

print(a.expression_raw)
print(a.expression)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

# print()
cfg = NestedDict(
        enu = NestedDict(
            bundle = 'dummy',
            name = 'enu',
            input = False,
            size = 10,
            debug = False
            ),
        epos = NestedDict(
            bundle = 'dummy',
            name = 'epos',
            input = False,
            size = 10,
            debug = False
            ),
        spec = NestedDict(
            bundle = 'dummy',
            name = 'spec',
            input = 1,
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

    graph = GNADot( context.outputs.enu, joints=False )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
