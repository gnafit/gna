#!/usr/bin/env python

from load import ROOT as R
from gna.expression import *
import numpy as np
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundles
from gna.env import env
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar, savefig
import gna.constructors as C
from gna.labelfmt import formatter as L

from gna.bindings import common

"""Parse arguments"""
#  from argparse import ArgumentParser
#  parser = ArgumentParser()
#  #  parser.add_argument( '-l', '--log', action='store_true', help='logarithmic scale' )
#  opts=parser.parse_args()

"""Init expression"""
indices = [
    ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241']),
     ('r', 'reactor', ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4'])
    ]

lib = dict(
)

expr = 'reactor_offeq[i,r]()'
a = Expression(expr, indices=indices)


print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        reactor_offeq = NestedDict(
            bundle = 'reactor_offeq_spectra_v02',
            offeq_data = './data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
            edges = np.linspace(0., 12, 240 + 1)
            ),
        )


"""Build"""
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )
import IPython; IPython.embed()
