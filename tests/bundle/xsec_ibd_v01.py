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

# lib = dict(
# )

expr = 'enu()'
a = Expression(expr)

# print(a.expression_raw)
# print(a.expression)

# a.parse()
# a.guessname(lib, save=True)
# a.tree.dump(True)

# print()
# cfg = NestedDict(
        # enu = NestedDict(
            # bundle = 'dummy',
            # name = 'enu',
            # input = False,
            # size = 10,
            # debug = False
            # ),
        # oscprob = NestedDict(
            # bundle = 'oscprob_v01',
            # name = 'oscprob',
            # input = True,
            # size = 10,
            # debug = False
            # ),
        # )
# context = ExpressionContext(cfg, ns=env.globalns)
# a.build(context)

# from gna.bindings import OutputDescriptor
# env.globalns.printparameters( labels=True )
# print( 'outputs:' )
# print( context.outputs )

if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot( context.outputs.enu, joints=False )
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
