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
weights = [ 'weight0', 'weight12neg', 'weight13neg', 'weight23neg' ]

ns = env.ns("")

from_nu = ROOT.Neutrino.ae()
to_nu = ROOT.Neutrino.ae()

gpu = True
#gpu = False

E_arr = N.arange(1.0, 10.0, 0.01 )  #array energy (МеV)
comp0 = N.array(N.ones(900))
E = Points(E_arr)
com = Points(comp0)
E.points.setLabel("Energy")
points1 = Points( arr1 )
points2 = Points( arr2 )
ndata=950
with context.manager(ndata) as manager:
    ns.defparameter("L", central=52,sigma=0) #kilometre

    gna.parameters.oscillation.reqparameters(ns)

    pmnsexpr = ROOT.OscProbPMNSExpressions(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(), ns=ns)
    ns.materializeexpressions()
    ns.printparameters()

    with ns:
        #Vacuum neutrino (same antineutrino)
        oscprob = C.OscProbPMNS(from_nu, to_nu)
        oscprob.comp12.inputs.Enu(E)
        if gpu:
            oscprob.comp12.switchFunction("gpu")
        data_osc = oscprob.comp12.comp12
        oscprob.comp12.setLabel('P | &#8710;m12')
        oscprob.comp13.inputs.Enu(E)
        if gpu:
            oscprob.comp13.switchFunction("gpu")
        data_osc2 = oscprob.comp13.comp13
        oscprob.comp13.setLabel('P | &#8710;m13')

        oscprob.comp23.inputs.Enu(E)
        if gpu:
            oscprob.comp23.switchFunction("gpu")
        data_osc3 = oscprob.comp23.comp23
        oscprob.comp23.setLabel('P | &#8710;m23')

        # Oscillation probability as weighted sum
        unity = C.FillLike(1, labels='Unity')
        unity.fill.setLabel(r'Unity')
        E >> unity.fill.inputs[0]

#        oscprob.compCP.inputs.Enu(E)
#        oscprob.compCP.switchFunction("gpu")
#        data_osc4 = oscprob.compCP.compCP

        #    env.printparameters()
        #    reqparameters(env.globalns)
        print('Create weighted sum')
        ws = C.WeightedSum( stdvector(weights), stdvector(labels) )
        ws.sum.item12(oscprob.comp12)
        ws.sum.item13(oscprob.comp13)
        ws.sum.item23(oscprob.comp23)
        ws.sum.comp0(unity.fill)
        if gpu:
            ws.sum.switchFunction("gpu")
        ws.sum.setLabel('OscProb')
        ns.materializeexpressions()
        pars = tuple(par.getVariable() for (name,par) in ns.walknames())
        manager.setVariables(C.stdvector(pars))
#va = manager.getVarArray()
#vaout = va.vararray.points



'''
print( 'Mode1: a1*w1+a2*w2' )
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )
'''
N=1000
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

    ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('Total time', elapsed_time)

print('GNA time (%i trials)'%N, elapsed_time-fake_time)
print('GNA time per event', (elapsed_time-fake_time)/N)

from gna.graphviz import GNADot

graph = GNADot( ws.sum )
graph.write("dotfile.dot")
plt.rcParams.update({'font.size': 22})  
plt.plot(E_arr, ws.sum.sum.data())
plt.title(r'$\overline{\nu}_e$ survival probability at 52 km')
plt.xlabel(r'$E_{\nu}$, MeV')
plt.ylabel(u'$P_{ee}$')
plt.grid()
plt.show()

