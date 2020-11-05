#!/usr/bin/env python

from gna.expression import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundles
from load import ROOT as R
from argparse import ArgumentParser
from gna.env import env
R.GNAObject

parser = ArgumentParser()
args = parser.parse_args()

indices = [
    ('r', 'reactor',    ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']),
    ('i', 'isotope',    ['U235', 'U238', 'Pu239', 'Pu241'])
    ]

lib = dict(
    reactor_info = dict( expr='reactor_info' )
)

expr = ['reactor_info[r, i]']
a = Expression(expr, indices=indices)

name_mapping = { idx+1: name for idx, name in enumerate(indices[0][-1])}

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        reactor_info = NestedDict(
            bundle = 'dayabay_reactor_burning_info_v01',
            reactor_info = 'data/dayabay/reactor/power/WeeklyAvg_P15A_v1.txt.npz',
            fission_uncertainty_info = 'data/dayabay/reactor/fission_fraction/2013.12.05_xubo.py'
            )
        )
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

