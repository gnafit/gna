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
from gna import context, bindings

"""Initialize inpnuts"""
arr1 = N.arange(0, 5)
arr2 = -arr1
print( 'Data1:', arr1 )
print( 'Data2:', arr2 )

labels = [ 'arr1', 'arr2' ]
weights = [ 'w1', 'w2' ]

ndata=7
with context.manager(ndata) as manager, context.cuda(enabled=True):
    """Initialize environment"""
    p1 = env.globalns.defparameter( weights[0], central= 1.0, sigma=0.1 )
    p2 = env.globalns.defparameter( weights[1], central= 1.0, sigma=0.1 )

    env.globalns.printparameters()
#    reqparameters(env.globalns)
    """Initialize transformations"""
    points1 = Points( arr1 )
    points2 = Points( arr2 )

    """Mode1: a1*w1+a2*w2"""
    print('Create weighted sum')
    ws = C.WeightedSum( stdvector(weights), stdvector(labels) )
    points1.points >> ws.sum.arr1
    points2.points >> ws.sum.arr2
#    pars = tuple(par.getVariable() for (name,par) in env.globalns.walknames())
#    manager.setVariables(C.stdvector([par.getVariable() for (name, par) in ns.walknames()]))
#    print('Set variables', pars)
#    manager.setVariables(C.stdvector(pars))
#    print('Variables set')

#va = manager.getVarArray()
#vaout = va.vararray.points
ws.sum.taint()
print( 'Mode1: a1*w1+a2*w2' )
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )

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
p2.set(-1)
print('  parameters', p1.value(), p2.value())
#print('  parameters memory block', vaout.data())
print('  result', ws.sum.sum.data() )

from gna.graphviz import savegraph
savegraph(ws.sum, "dotfile.dot")

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
