#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundles
from matplotlib import pyplot as plt, gridspec as gs
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import gna.constructors as C
import numpy as np
from gna.configurator import NestedDict, uncertain
from physlib import percent
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.bindings import common

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()
args.make_idx=True

km = 1000.0
L1, L2 = 52*km, 100*km
detectors = {
    'AD1' : [ L1, 0.0, 0.0 ],
    'AD2' : [ L2, 0.0, 0.0 ],
}
reactors = {
    'Core0' : [0.0, 0.0, 0.0],
}
# components

indices = [('r', 'reactor', list(reactors.keys()) ),
           ('d', 'detector', list(detectors.keys()) ),
           ('c', 'component', ['comp0', 'comp12', 'comp13', 'comp23'])
           ]

expr = [
        "baseline[r,d]()",
        "sum[c]| pmns[c]*oscprob[c,d,r](enu())",
        "oscprob_matter[d,r](enu())"
]

a =  Expression_v01(expr, indices=indices)
a.parse()
lib = dict(
        oscprob_comp = dict(expr='oscprob*pmns'),
        oscprob_full = dict(expr='sum:c|oscprob_comp'),
        )
a.guessname(lib, save=True)
a.tree.dump(True)

enu = C.Points(np.arange(1.0, 12.0+1.e-9, 0.001))
enu_o = enu.single()
#
# Initialize bundles
#
cfg_idx = NestedDict(
        baselines = NestedDict(
            # Bundle name
            bundle = dict(
                name = 'reactor_baselines', version = 'v01',
                major = 'rd', # first for reactor, second for detector
                ),
            # Reactor positions
            reactors = reactors,
            # Detector positions
            detectors = detectors,
            unit='meter'
            ),
        oscprob = NestedDict(
            bundle = dict(name='oscprob', version='v04', major='rdc'),
            pdgyear = 2016
            ),
        oscprob_matter = NestedDict(
            bundle = dict(name='oscprob_matter', version='v01', major='rd',
                          names=dict(pmns='pmns_matter', oscprob='oscprob_matter')),
            density = 2.6, # g/cm3
            pdgyear = 2016
            ),
        enu = NestedDict(
            bundle = NestedDict(name='predefined', version='v01', major=''),
            name = 'enu',
            inputs = None,
            outputs = enu.single(),
            ),
        )
context = ExpressionContext_v01(cfg_idx, ns=env.globalns)
a.build(context)

env.globalns.printparameters(labels=True)

#
# Test bundle
#
def plot(vacuum, matter, title):
    fig = plt.figure()
    grid = gs.GridSpec(3, 1, hspace=0)
    ax = plt.subplot(grid[:2], xlabel=None, ylabel='P', title=title)
    ax.minorticks_on()
    ax.grid()

    vacuum.plot_vs(       enu_o, linestyle='dashed', alpha=0.6, label='Vacuum')
    matter.plot_vs(enu_o, linestyle='dotted', alpha=0.6, label='Matter')
    ax.legend()

    ax2 = plt.subplot(grid[2], xlabel='Enu', ylabel='reldiff', sharex=ax)
    ax2.minorticks_on()
    ax2.grid()

    data_vac = vacuum.data()
    data_mat = matter.data()
    reldif = data_vac/data_mat - 1.0
    ax2.plot(enu_o.data(), reldif)


for (vacuum, matter, L) in zip(context.outputs.oscprob_full.Core0.values(), context.outputs.oscprob_matter.Core0.values(), (L1, L2)):
    plot(vacuum, matter, title='Oscillation probability comparison, L={} km'.format(L/km))

plt.show()
