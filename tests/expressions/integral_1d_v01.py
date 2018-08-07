#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundle
from load import ROOT as R
from argparse import ArgumentParser
from gna.env import env
from matplotlib import pyplot as P
import numpy as N
from mpl_tools import bindings
R.GNAObject

parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
args = parser.parse_args()

indices = [
        ('k', 'kin', ['a', 'b', 'c'])
        ]
lib = dict(
        scaled_e = dict(expr='weight*evis'),
)

pars = tuple()
pars+=env.globalns.defparameter('weight.a', central=1, sigma=0.1),
pars+=env.globalns.defparameter('weight.b', central=2, sigma=0.1),
pars+=env.globalns.defparameter('weight.c', central=3, sigma=0.1),

expr = 'kinint| weight[k] * evis()'
a = Expression(expr, indices)

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        kinint = NestedDict(
            bundle   = 'integral_1d_v01',
            variable = 'evis',
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            orders   = 3,
            provides = [ 'evis' ]
            ),
        )
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'E' )
ax.set_ylabel( 'int f(E)' )
ax.set_title( 'Linear function f(E)=E integrated' )

edges = cfg.kinint.edges
centers = (edges[1:]+edges[:-1])*0.5
widths  = (edges[1:]-edges[:-1])
ints    = centers*widths
for i, (par, out) in enumerate(zip(pars, context.outputs.kinint.values())):
    out.plot_hist(label='GL quad '+str(i))
    ax.plot( centers, ints*par.value(), '--', label='calc '+str(i) )

ax.legend(loc='upper left')

if args.show:
    P.show()

if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot(context.outputs.evis, joints=False)
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise
