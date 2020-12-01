#!/usr/bin/env python

from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.configurator import uncertaindict
from gna.bundle import execute_bundles
from load import ROOT as R
from argparse import ArgumentParser
from gna.env import env
from matplotlib import pyplot as P
import numpy as N
from gna.configurator import NestedDict
from mpl_tools import bindings
R.GNAObject

parser = ArgumentParser()
parser.add_argument( '--dot', nargs='+', default=(), help='write graphviz output' )
args = parser.parse_args()

indices = [
        ('k', 'kin', ['a', 'b', 'c'])
        ]
lib = dict(
        scaled_e = dict(expr='weight*evis'),
)

expr = 'integral[k]'
a = Expression_v01(expr, indices)

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        kinint = NestedDict(
            bundle   = dict(name='integral_1d_v02'),
            variable = 'evis',
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            orders   = 3,
            labels   = dict(
                sampler = 'GL Sampler',
                integrator = 'Integrator {autoindex}'
                )
            ),
        )
context = ExpressionContext_v01(cfg, ns=env.globalns)
a.build(context)

for fname in args.dot:
    from gna.graphviz import savegraph
    savegraph([context.outputs.evis]+list(context.outputs.integral.values()), fname)

