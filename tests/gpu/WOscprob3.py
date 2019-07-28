#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
import matplotlib.pyplot as plt
from load import ROOT
from gna import constructors as C
from gna.env import env
import gna.parameters
import gna.parameters.oscillation
from gna import context, bindings
import ROOT
import time

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--graph' )
parser.add_argument( '-p', '--precision', default='double', choices=['float', 'double'] )
parser.add_argument( '-g', '--gpu', action='store_true' )
args = parser.parse_args()
print('Mode:', args.gpu and 'GPU' or 'CPU')
print('Precision:', args.precision)

ROOT.GNAObject

labels = [ 'comp0', 'item12', 'item13','item23' ]
weights = [ 'weight0', 'weight12', 'weight13', 'weight23' ]
ns = env.ns("")

from_nu = ROOT.Neutrino.ae()
to_nu = ROOT.Neutrino.ae()

ndata=950
modecos = True # default

clabels = [ 'P | &#8710;m12', 'P | &#8710;m13', 'P | &#8710;m23' ]
E_arr = N.arange(1.0, 10.0, 0.001)  #array energy (МеV)

with context.set_context(manager=ndata, gpu=args.gpu, precision=args.precision) as manager:
    ns.defparameter("L", central=52,sigma=0) #kilometre
    gna.parameters.oscillation.reqparameters(ns)
    pmnsexpr = C.OscProbPMNSExpressions(from_nu, to_nu, modecos, ns=ns)
    ns.materializeexpressions()
    ns.printparameters()

    E = C.Points(E_arr, labels='Energy')

    with ns:
        oscprob = C.OscProb3(from_nu, to_nu, 'L', modecos, labels=clabels)
        unity = C.FillLike(1, labels='Unity')

        E >> (unity.fill, oscprob.comp12, oscprob.comp13, oscprob.comp23)

        ws = C.WeightedSum(weights, labels, labels='OscProb')
        unity          >> ws.sum.comp0
        oscprob.comp12 >> ws.sum.item12
        oscprob.comp13 >> ws.sum.item13
        oscprob.comp23 >> ws.sum.item23

        ns.materializeexpressions()
        pars = tuple(par.getVariable() for (name,par) in ns.walknames())
        manager.setVariables(C.stdvector(pars))

N=100
start_time = time.time()
for x in range(N):
    ws.sum.taint()
    oscprob.comp12.taint()
    oscprob.comp13.taint()
    oscprob.comp23.taint()

end_time = time.time()
fake_time = end_time - start_time
print('Fake time', fake_time)

start_time = time.time()
for x in range(N):
    ws.sum.taint()
    oscprob.comp12.taint()
    oscprob.comp13.taint()
    oscprob.comp23.taint()

    # ws.sum.sum.data()
    ws.sum.touch_global()
end_time = time.time()
elapsed_time = end_time - start_time
print('Total time', elapsed_time)

print('GNA time (%i trials)'%N, elapsed_time-fake_time)
print('GNA time per event', (elapsed_time-fake_time)/N)

if args.graph:
    from gna.graphviz import savegraph
    savegraph(ws.sum, args.graph)

    name, ext = args.graph.rsplit('.', 1)
    savegraph(ws.sum, name+'_vars.'+ext, namespace=ns)


from gna.bindings import common
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_title(r'$\overline{\nu}_e$ survival probability at 52 km')
ax.set_xlabel(r'$E_{\nu}$, MeV')
ax.set_ylabel(u'$P_{ee}$')
ws.sum.sum.plot_vs(E_arr)

print(ws.sum.sum)
plt.show()

