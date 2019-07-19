#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
import matplotlib.pyplot as plt
from load import ROOT
from gna import constructors as C # construct objects
from gna.constructors import Points, stdvector
from gna.env import env
import gna.parameters
import gna.parameters.oscillation
from gna import context, bindings
import ROOT
import time

ROOT.GNAObject

"""Initialize inpnuts"""
arr1 = N.arange(1, 1001)
arr2 = -arr1
#print( 'Data1:', arr1 )
#print( 'Data2:', arr2 )

labels = [ 'comp0', 'item12', 'item13','item23' ]
weights = [ 'weight0', 'weight12', 'weight13', 'weight23' ]

ns = env.ns("")

from_nu = ROOT.Neutrino.ae()
to_nu = ROOT.Neutrino.ae()

gpu = False

E_arr = N.arange(1.0, 10.0, 0.001)  #array energy (МеV)
E = Points(E_arr)
E.points.setLabel("Energy")
points1 = Points( arr1 )
points2 = Points( arr2 )
ndata=950
with context.set_context(manager=ndata, gpu=gpu) as manager:
    ns.defparameter("L", central=52,sigma=0) #kilometre

    gna.parameters.oscillation.reqparameters(ns)

    pmnsexpr = ROOT.OscProbPMNSExpressions(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(), ns=ns)
    ns.materializeexpressions()
    ns.printparameters()

    clabels = [ 'P | &#8710;m12', 'P | &#8710;m13', 'P | &#8710;m23' ]
    with ns:
        oscprob = C.OscProb3(from_nu, to_nu, labels=clabels)
        unity = C.FillLike(1, labels='Unity')

        E >> (unity.fill, oscprob.comp12, oscprob.comp13, oscprob.comp23)

        ws = C.WeightedSum(stdvector(weights), stdvector(labels), labels='OscProb')
        unity          >> ws.sum.comp0
        oscprob.comp12 >> ws.sum.item12
        oscprob.comp13 >> ws.sum.item13
        oscprob.comp23 >> ws.sum.item23

        ns.materializeexpressions()
        pars = tuple(par.getVariable() for (name,par) in ns.walknames())
        manager.setVariables(C.stdvector(pars))

# N=1000
# start_time = time.time()
# for x in range(N):
    # ws.sum.taint()
    # oscprob.comp12.taint()
    # oscprob.comp13.taint()
    # oscprob.comp23.taint()

# end_time = time.time()
# fake_time = end_time - start_time
# print('Fake time', fake_time)

# start_time = time.time()
# for x in range(N):
    # ws.sum.taint()
    # oscprob.comp12.taint()
    # oscprob.comp13.taint()
    # oscprob.comp23.taint()

    # ws.sum.sum.data()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print('Total time', elapsed_time)

# print('GNA time (%i trials)'%N, elapsed_time-fake_time)
# print('GNA time per event', (elapsed_time-fake_time)/N)

from gna.graphviz import savegraph
savegraph(ws.sum, "output/oscprob3.dot")

from gna.bindings import common
fig = plt.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.title(r'$\overline{\nu}_e$ survival probability at 52 km')
ax.xlabel(r'$E_{\nu}$, MeV')
ax.ylabel(u'$P_{ee}$')
ws.sum.sum.plot_vs(E_arr)
plt.show()

