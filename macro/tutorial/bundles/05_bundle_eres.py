#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna.env import env
from gna.bindings import common
from gna import constructors as C
import numpy as np
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig

cfg = NestedDict(
    bundle = dict(
        name='detector_eres',
        version='ex01',
        ),
    parameter = 'eres',
    pars = uncertaindict(
        [
         ('a', 0.01),
         ('b', 0.09),
         ('c', 0.03),
         ],
        mode='percent',
        uncertainty = 30.0
        ),
)
b = execute_bundle(cfg)
env.globalns.printparameters(labels=True); print()

from sys import argv
oname = 'output/tutorial/'+argv[0].rsplit('/', 1).pop().replace('.py', '')

#
# Prepare inputs
#
emin, emax = 0.0, 12.0
nbins = 240
edges = np.linspace(emin, emax, nbins+1, dtype='d')
data  = np.zeros(nbins, dtype='d')
data[20]=1.0  # 1 MeV
data[120]=1.0 # 6 MeV
data[200]=1.0 # 10 MeV
hist = C.Histogram(edges, data)
hist.hist.setLabel('Input histogram')

edges_o = C.Points(edges)
edges_o.points.setLabel('Bin edges')

# Bind outputs
#
edges_o >> b.context.inputs.eres_matrix.values()
hist    >> b.context.inputs.eres.values()
print( b.context )

from gna.graphviz import savegraph
savegraph(hist, oname+'_graph.png')

#
# Plot
#
fig = plt.figure()
ax = plt.subplot(111, xlabel='E, MeV', ylabel='', title='Energy smearing')
ax.minorticks_on()
ax.grid()

hist.hist.hist.plot_hist(label='Original histogram')
b.context.outputs.eres.plot_hist(label='Smeared histogram')

ax.legend(loc='upper right')

savefig(oname+'.png')
plt.show()
