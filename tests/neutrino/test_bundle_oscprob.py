#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundles
from matplotlib import pyplot as P
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

detectors = {
    'AD1' : [ 52000,  0.0, 0.0 ],
    'AD2' : [ 100000, 0.0, 0.0 ],
}
reactors = {
    'Core0' : [0.0, 0.0, 0.0],
}
# components

indices = [('r', 'reactor', list(reactors.keys()) ),
           ('d', 'detector', list(detectors.keys()) ),
           ('c', 'component', ['comp0', 'comp12', 'comp13', 'comp23'])
           ]

expr = '''
        baseline[r,d]()
        sum[c]| pmns[c]*oscprob[c,d,r](enu())
        sum[c]| pmns_matter[c]*oscprob_matter[c,d,r](enu())
'''

a =  Expression_v01(expr, indices=indices)
a.parse()
lib = dict(
        oscprob_comp = dict(expr='oscprob*pmns'),
        oscprob_full = dict(expr='sum:c|oscprob_comp'),
        oscprob_comp_matter = dict(expr='oscprob_matter*pmns_matter'),
        oscprob_full_matter = dict(expr='sum:c|oscprob_comp_matter'),
        )
a.guessname(lib, save=True)
a.tree.dump(True)

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
            bundle = dict(name='oscprob', version='v04', major='rdc',
                          names=dict(pmns='pmns_matter', oscprob='oscprob_matter')),
            pdgyear = 2016
            ),
        enu = NestedDict(
            bundle = NestedDict(name='predefined', version='v01', major=''), name = 'enu',
            inputs = None,
            outputs = C.Points(np.arange(0.0, 12.0, 0.001)),
            ),
        )
context = ExpressionContext_v01(cfg_idx, ns=env.globalns)
a.build(context)

env.globalns.printparameters(labels=True)

#
# Test bundle
#
fig = P.figure()
ax = P.subplot(111, xlabel='Enu', ylabel='P', title='Oscillation probability comparison')
ax.minorticks_on()
ax.grid()


