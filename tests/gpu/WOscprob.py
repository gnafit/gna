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

labels = [ 'item12', 'item13','item23', 'comp0' ]
weights = [ 'weight12neg', 'weight13neg', 'weight23neg', 'comp0w' ]

ns = env.ns("")

from_nu_a = ROOT.Neutrino.amu()
to_nu_a = ROOT.Neutrino.ae()

from_nu = ROOT.Neutrino.mu()
to_nu = ROOT.Neutrino.e()

E_arr = N.arange(1.0, 10.0, 0.01)  #array energy (МеV)
comp0 = N.array(N.ones(900))
E = Points(E_arr)
com = Points(comp0)
E.points.setLabel("Energy")
points1 = Points( arr1 )
points2 = Points( arr2 )
ndata=950
with context.manager(ndata) as manager:
    ns.defparameter("L", central=52,sigma=0) #kilometre
    ns.defparameter("comp0w", central=1.0,sigma=0) #kilometre

    gna.parameters.oscillation.reqparameters(ns)

    pmnsexpr = ROOT.OscProbPMNSExpressions(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(), ns=ns)
    ns.materializeexpressions()
    ns.printparameters()

    with ns:
        #Vacuum neutrino (same antineutrino)
        oscprob = C.OscProbPMNS(from_nu, to_nu)
        oscprob.comp12.inputs.Enu(E)
        oscprob.comp12.switchFunction("gpu")
        data_osc = oscprob.comp12.comp12
        oscprob.comp12.setLabel('P | &#8710;m12')
        oscprob.comp13.inputs.Enu(E)
        oscprob.comp13.switchFunction("gpu")
        data_osc2 = oscprob.comp13.comp13
        oscprob.comp13.setLabel('P | &#8710;m13')

        oscprob.comp23.inputs.Enu(E)
        oscprob.comp23.switchFunction("gpu")
        data_osc3 = oscprob.comp23.comp23
        oscprob.comp23.setLabel('P | &#8710;m23')

        # Oscillation probability as weighted sum
        unity = C.FillLike(1, labels='Unity')
        E >> unity.fill.inputs[0]
        unity.fill.setLabel('comp0')
	
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
for x in range(0,20):
    ws.sum.taint()
    oscprob.comp12.taint()
    oscprob.comp13.taint()
    oscprob.comp23.taint()
    
    start_time = time.time()
    ws.sum.sum.data()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    


from gna.graphviz import GNADot

graph = GNADot( ws.sum )
graph.write("dotfile.dot")

plt.plot(E_arr*1e-3, ws.sum.sum.data())
plt.xlabel('Energy')
plt.ylabel(r'$P_{\nu_{\mu} \to \nu_{e}}$')
plt.grid()
plt.show()

