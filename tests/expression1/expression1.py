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
    ('n', 'num',   ['1', '2']),
    ('a', 'alph',  ['a', 'b']),
    ('z', 'zyx',   ['X', 'Y']),
    ('b', 'bkg',   ['b1', 'b2'])
    ]

lib = dict(
    obs_spec = dict(expr='norm*spec'),
    obs_tot  = dict(expr='obs_spec+bkg'),
    insum    = dict(expr='sum:a_b'),
    inproduct= dict(expr='prod:n'),
    totalsum = dict(expr='sum:z')
)

expr = 'sum[z] | prod[n] | sum[a,b] | norm()*spec[n](enu[a,z]()) + bkg[b]()'
a = Expression(expr, indices=indices)

print(a.expression_raw)
print(a.expression)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
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
        norm = NestedDict(
            bundle = 'dummy',
            name = 'norm',
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
        bkg1 = NestedDict(
            bundle = 'dummy',
            name = 'bkg',
            indices = [ ('b', 'bkg',   ['b1']) ],
            input = False,
            size = 10,
            debug = False,
            provides = [ 'bkg.b1' ]
            ),
        bkg2 = NestedDict(
            bundle = 'dummy',
            name = 'bkg',
            indices = [ ('b', 'bkg',   ['b2']) ],
            input = False,
            size = 10,
            debug = False,
            provides = [ 'bkg.b2' ]
            ),
        bkg3 = NestedDict(
            bundle = 'dummy',
            name = 'bkg',
            indices = [ ('b', 'bkg',   ['b3']) ],
            input = False,
            size = 10,
            debug = False,
            provides = [ 'bkg.b3' ]
            ),
        )
context = ExpressionContext( cfg )
a.build(context)

if args.dot:
    # try:
    from gna.graphviz import GNADot

    graph = GNADot( context.outputs.totalsum )
    graph.write(args.dot)
    print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
