#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from load import ROOT
from gna.expression import *
from gna.configurator import uncertaindict
from gna.bundle import execute_bundles
from argparse import ArgumentParser
from gna.env import env
ROOT.GNAObject

parser = ArgumentParser()
parser.add_argument('-s', '--show', action="store_true", help='Show the plot')
args = parser.parse_args()

indices = [("i", "isotopes", ["snf"])]

lib = dict(
    snf = dict( expr='spent_nuclear_fuel' )
)

expr = ['snf']
a = Expression(expr, indices=indices)

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)
edges=np.linspace(0., 12, 240+1)

print()
cfg = NestedDict(
    snf = NestedDict(
        bundle="reactor_snf_spectra_v02",
        data_path="data/reactor_anu_spectra/SNF/kopeikin_0412.044_spent_fuel_spectrum_smooth.dat",
        edges=edges)
    )
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

if args.show:
    import matplotlib.pyplot as plt
    import mpl_tools.helpers as mplh

    snf = context.outputs['snf_ratio']
    bin_centers = (edges[1:] + edges[:-1])/2
    mplh.plot_hist(edges, snf.data())
    plt.show()

