#!/usr/bin/env python

from tutorial import tutorial_image_name, savefig, savegraph
import load
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna import constructors as C
from gna.env import env
import numpy as np
from matplotlib import pyplot as plt
from gna.bindings import common

cfg = NestedDict(
    bundle = dict(
        name='detector_eres',
        version='ex02',
        nidx=[ ('d', 'detector', ['D1', 'D2', 'D3']),
               ('z', 'zone', ['z1', 'z2'])],
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
)
b = execute_bundle(cfg)
env.globalns.printparameters(labels=True); print()

#
# Prepare inputs
#
emin, emax = 0.0, 12.0
nbins = 240
edges = np.linspace(emin, emax, nbins+1, dtype='d')
data1  = np.zeros(nbins, dtype='d')
data1[20]=1.0  # 1 MeV
data1[120]=1.0 # 6 MeV
data1[200]=1.0 # 10 MeV

data2  = np.zeros(nbins, dtype='d')
data2[40]=1.0  # 2 MeV
data2[140]=1.0 # 7 MeV
data2[220]=1.0 # 11 MeV

data3  = np.zeros(nbins, dtype='d')
data3[60]=1.0  # 3 MeV
data3[160]=1.0 # 8 MeV
data3[239]=1.0 # 12 MeV

hist1 = C.Histogram(edges, data1)
hist1.hist.setLabel('Input histogram 1')

hist2 = C.Histogram(edges, data2)
hist2.hist.setLabel('Input histogram 2')

hist3 = C.Histogram(edges, data3)
hist3.hist.setLabel('Input histogram 3')

#
# Bind outputs
#
suffix = '' if cfg.split_transformations else 'merged_'
savegraph(b.context.outputs.smearing_matrix.values(), tutorial_image_name('png', suffix=suffix+'graph0'), rankdir='TB')

hist1   >> b.context.inputs.smearing_matrix.values(nested=True)
hist1   >> b.context.inputs.eres.D1.values(nested=True)
hist2   >> b.context.inputs.eres.D2.values(nested=True)
hist3   >> b.context.inputs.eres.D3.values(nested=True)
print( b.context )

savegraph(hist1, tutorial_image_name('png', suffix=suffix+'graph1'), rankdir='TB')

#
# Plot
#
fig = plt.figure(figsize=(12,12))

hists = [hist1, hist2, hist3]
for i, det in enumerate(['D1', 'D2', 'D3']):
    ax = plt.subplot(221+i, xlabel='E, MeV', ylabel='', title='Energy smearing in '+det)
    ax.minorticks_on()
    ax.grid()

    hists[i].hist.hist.plot_hist(label='Original histogram')
    for i, out in enumerate(b.context.outputs.eres[det].values(nested=True)):
        out.plot_hist(label='Smeared histogram (%i)'%i)

    ax.legend(loc='upper right')

savefig(tutorial_image_name('png'))
plt.show()
