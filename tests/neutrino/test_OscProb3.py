#!/usr/bin/env python

"""Check the WeightedSum transformation"""

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
args = parser.parse_args()

ROOT.GNAObject

labels = [ 'comp0', 'item12', 'item13','item23' ]
weights = [ 'weight0', 'weight12', 'weight13', 'weight23' ]
ns = env.ns("")

from_nu = ROOT.Neutrino.ae()
to_nu = ROOT.Neutrino.ae()

modecos = False # default

clabels = [ 'P | &#8710;m12', 'P | &#8710;m13', 'P | &#8710;m23' ]
E_arr = N.arange(1.0, 10.0, 0.001)  #array energy (МеV)

ns.defparameter("L", central=52,sigma=0) #kilometre
gna.parameters.oscillation.reqparameters_reactor(ns, dm='23')
pmnsexpr = C.OscProbPMNSExpressions(from_nu, to_nu, modecos, ns=ns)
ns.materializeexpressions()
ns.printparameters(labels=True)

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

if args.graph:
    from gna.graphviz import savegraph
    savegraph(ws.sum, args.graph)

    name, ext = args.graph.rsplit('.', 1)
    savegraph(ws.sum, name+'_vars.'+ext, namespace=ns)

out = ws.sum.sum

from gna.bindings import common
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_title(r'$\overline{\nu}_e$ survival probability at 52 km')
ax.set_xlabel(r'$E_{\nu}$, MeV')
ax.set_ylabel(u'$P_{ee}$')
out.plot_vs(E_arr, label='Full')

ns['SinSqDouble13'].push(0.0)
out.plot_vs(E_arr, label=r'Only solar: $\sin^22\theta_{13}=0$')
print('Solar mode')
ns.printparameters(labels=True)

ax.legend()
print(ws.sum.sum)
plt.show()

