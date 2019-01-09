#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna import constructors as C
from gna.env import env
import numpy as np
from matplotlib import pyplot as plt
from gna.bindings import common
from gna.graphviz import savegraph
from mpl_tools.helpers import savefig

#
# Prepare inputs
#
emin, emax = 0.0, 12.0
nbins = 240
edges = np.linspace(emin, emax, nbins+1, dtype='d')
peaks = 20, 120, 200
hists = ()

for i in range(3):
    data  = np.zeros(nbins, dtype='d')

    for peak in peaks:
        pos = np.max((peak+20*i, nbins-1))
        data[pos]=1.0

    hist = C.Histogram(edges, data)
    hist.hist.setLabel('Input histogram %i'%i)
    hists += hist,

cfg = NestedDict(
        bundle = dict(
            name = 'expression',
            version = 'v01',
            nidx=[ ('d', 'detector', ['D1', 'D2', 'D3']),
                   ('z', 'zone', ['z1', 'z2'])],
            ),
        verbose = 2,

        # Expression
        expr = 'norm1[d] * norm2[z] * eres[z,d]| hist[z]()',

        # Configuration
        bundles = NestedDict(
            eres = NestedDict(
                bundle = dict(
                    name='detector_eres_normal',
                    version='v01',
                    major=['z'],
                    names=dict(
                        eres_matrix='smearing_matrix',
                        )
                    ),
                parameter = 'eres',
                pars = uncertaindict(
                    [
                     ('z1.a', (0.0, 'fixed')),
                     ('z1.b', (0.05, 30, 'percent')),
                     ('z1.c', (0.0, 'fixed')),
                     ('z2.a', (0.0, 'fixed')),
                     ('z2.b', (0.10, 30, 'percent')),
                     ('z2.c', (0.0, 'fixed')),
                     ('z3.a', (0.0, 'fixed')),
                     ('z3.b', (0.15, 30, 'percent')) ,
                     ('z3.c', (0.0, 'fixed')),
                     ]
                    ),
                labels = dict(
                    matrix    = 'Smearing\nmatrix\n{autoindex}',
                    smear     = 'Energy\nresolution\n{autoindex}',
                    parameter = '{description} (zone {autoindex})'
                    ),
                split_transformations = True
            ),
            norm1 = NestedDict(
                bundle = 'parameters_v01',
                parameter = 'norm1',
                label='Normalization at detector {detector}',
                pars = uncertaindict(
                    [
                        ( 'D1', 1.0 ),
                        ( 'D2', 2.0 ),
                        ( 'D3', 3.0 ),
                        ],
                    uncertainty = 1.0,
                    mode = 'percent',
                    ),
            ),
            norm2 = NestedDict(
                bundle = 'parameters_v01',
                parameter = 'norm2',
                label='Normalization at zone {zone}',
                pars = uncertaindict(
                    [
                        ( 'z1', 1.0 ),
                        ( 'z2', 1.1 ),
                        ( 'z3', 1.2 ),
                        ],
                    uncertainty = None,
                    mode = 'fixed',
                    ),
            ),
            input = NestedDict(
                    bundle = 'predefined_v01',
                    name = 'hist',
                    inputs = None,
                    outputs = NestedDict(
                        'z1' = hists[0],
                        'z2' = hists[1],
                        'z3' = hists[2]
                        )
                    )
        ),

        # Name comprehension
        lib = dict(
                norm = dict( expr='norm1*norm2' )
                )
)

b = execute_bundle(cfg)
env.globalns.printparameters(labels=True); print()

print('Inputs')
print(b.context.inputs)

print('Outputs')
print(b.context.outputs)

print('Parameters')
b.namespace.printparameters

# from sys import argv
# oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '')

# #
# # Bind outputs
# #
# suffix = '' if cfg.split_transformations else '_merged'
# savegraph(b.context.outputs.smearing_matrix.values(), oname+suffix+'_graph0.png')

# hist1   >> b.context.inputs.smearing_matrix.values(nested=True)
# hist1   >> b.context.inputs.eres.D1.values(nested=True)
# hist2   >> b.context.inputs.eres.D2.values(nested=True)
# hist3   >> b.context.inputs.eres.D3.values(nested=True)
# print( b.context )

# savegraph(hist1, oname+suffix+'_graph1.png')

# #
# # Plot
# #
# fig = plt.figure(figsize=(12,12))

# hists = [hist1, hist2, hist3]
# for i, det in enumerate(['D1', 'D2', 'D3']):
    # ax = plt.subplot(221+i, xlabel='E, MeV', ylabel='', title='Energy smearing in '+det)
    # ax.minorticks_on()
    # ax.grid()

    # hists[i].hist.hist.plot_hist(label='Original histogram')
    # for i, out in enumerate(b.context.outputs.eres[det].values(nested=True)):
        # out.plot_hist(label='Smeared histogram (%i)'%i)

    # ax.legend(loc='upper right')

# savefig(oname+'.png')
# plt.show()
