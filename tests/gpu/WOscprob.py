#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
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

labels = [ 'arr1', 'arr2','arr3' ]
weights = [ 'w1', 'w2', 'w3' ]


ns = env.ns("")

from_nu_a = ROOT.Neutrino.amu()
to_nu_a = ROOT.Neutrino.ae()

from_nu = ROOT.Neutrino.mu()
to_nu = ROOT.Neutrino.e()

E_arr = N.array(range(1, 241, 1))  #array energy (МеV)
#comp0 = N.array(N.ones(1000))
E = Points(E_arr)
#com = Points(comp0)

points1 = Points( arr1 )
points2 = Points( arr2 )
ndata=260
with context.manager(ndata) as manager:
    gna.parameters.oscillation.reqparameters(ns)

    ns.defparameter("L", central=810,sigma=0) #kilometre
    ns.defparameter("rho",central=2.7,sigma=0) #g/cm3

    """Initialize environment"""
    p1 = ns.defparameter( weights[0], central= 1.0, sigma=0.1 )
    p2 = ns.defparameter( weights[1], central= 1.0, sigma=0.1 )
    p3 = ns.defparameter( weights[2], central= 1.0, sigma=0.1 )

    with ns:
        #Vacuum neutrino (same antineutrino)
        oscprob = C.OscProbPMNS(from_nu, to_nu)
        oscprob.comp12.inputs.Enu(E)
        oscprob.comp12.switchFunction("gpu")
        data_osc = oscprob.comp12.comp12

        oscprob.comp13.inputs.Enu(E)
        oscprob.comp13.switchFunction("gpu")
        data_osc2 = oscprob.comp13.comp13

        oscprob.comp23.inputs.Enu(E)
        oscprob.comp23.switchFunction("gpu")
        data_osc3 = oscprob.comp23.comp23

#        oscprob.compCP.inputs.Enu(E)
#        oscprob.compCP.switchFunction("gpu")
#        data_osc4 = oscprob.compCP.compCP

        #    env.printparameters()
        #    reqparameters(env.globalns)
        print('Create weighted sum')
        ws = C.WeightedSum( stdvector(weights), stdvector(labels) )
        ws.sum.arr1(oscprob.comp12)
        ws.sum.arr2(oscprob.comp13)
        ws.sum.arr3(oscprob.comp23)
        ws.sum.switchFunction("gpu")
        ns.materializeexpressions()
        pars = tuple(par.getVariable() for (name,par) in ns.walknames())
        manager.setVariables(C.stdvector(pars))
'''
    with ns:
        #Vacuum neutrino (same antineutrino)
        oscprob = C.OscProbPMNS(from_nu, to_nu)
        oscprob.comp12.inputs.Enu(E)
        oscprob.comp12.switchFunction("gpu")
        data_osc = oscprob.comp12.comp12

        oscprob.comp13.inputs.Enu(E)
        oscprob.comp13.switchFunction("gpu")
        data_osc2 = oscprob.comp13.comp13

        oscprob.comp23.inputs.Enu(E)
        oscprob.comp23.switchFunction("gpu")
        data_osc3 = oscprob.comp23.comp23

        oscprob.compCP.inputs.Enu(E)
        oscprob.compCP.switchFunction("gpu")
        data_osc4 = oscprob.compCP.compCP
'''
#va = manager.getVarArray()
#vaout = va.vararray.points



'''
print( 'Mode1: a1*w1+a2*w2' )
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )
'''
start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()


start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()

start_time = time.time()
ws.sum.sum.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('time = ', elapsed_time)

ws.sum.taint()
oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()



'''
print()
p1.set(2)
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )

print()
p2.set(2)
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )

print()
p1.set(1)
p2.set(-1.5)
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )
'''

from gna.graphviz import GNADot

graph = GNADot( ws.sum )
graph.write("dotfile.dot")

#print('  result', ws.sum.sum.data() )

"""
""" """Mode2: a1*w1+a2""" """
with context.manager(ndata) as manager:
    ws = C.WeightedSum( stdvector(weights[:1]), stdvector(labels) )
    ws.sum.arr1(points1.points)
    ws.sum.arr2(points2.points)
    ws.sum.switchFunction("gpu")

print( 'Mode2: a1*w1+a2' )
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p2.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(1)
p2.set(1)
print()

""" """Mode4: c+a1*w1+a1*w2""" """
with context.manager(ndata) as manager:
    ws = C.WeightedSum( -10, stdvector(weights), stdvector(labels) )
    ws.sum.arr1(points1.points)
    ws.sum.arr2(points2.points)
    ws.sum.switchFunction("gpu")

print( 'Mode4: -10+a1*w1+a2*w2' )
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p2.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(1)
p2.set(1)
print()
"""
