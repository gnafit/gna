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
args = parser.parse_args()

indices = [
    ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']),
    ]

lib = dict(
)

expr = 'eff * fcn()'
a = Expression(expr, indices=indices)

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        eff = NestedDict(
            bundle = 'efficiencies_v01',
            ),
        fcn = NestedDict(
            bundle = 'dummy',
            name = 'dymmubundle',
            size = 10,
            debug = False
            )
        )
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

