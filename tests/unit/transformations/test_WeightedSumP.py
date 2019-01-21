#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.constructors import Points, VarArray, WeightedSumP
from gna.env import env

"""Initialize inpnuts"""
arr1 = N.arange(0, 5, dtype='d')
arr2 = -arr1
zeros = N.zeros((5,), dtype='d')
print( 'Data1:', arr1 )
print( 'Data2:', arr2 )

labels = [ 'arr1', 'arr2' ]
weights = [ 'w1', 'w2' ]

"""Initialize environment"""
p1 = env.globalns.defparameter( weights[0], central=1.0, sigma=0.1 )
p2 = env.globalns.defparameter( weights[1], central=1.0, sigma=0.1 )

env.globalns.printparameters()

pp1 = VarArray([weights[0]])
pp2 = VarArray([weights[1]])

"""Initialize transformations"""
points1 = Points( arr1 )
points2 = Points( arr2 )

def test_01():
    outputs=[o.single() for o in [pp1, points1, pp2, points2]]
    ws = WeightedSumP(outputs)

    ws.print()

    print( 'Mode1: a1*w1+a2*w2' )
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==zeros).all()
    p1.set(2)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==arr1).all()
    p2.set(2)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==zeros).all()
    p1.set(1)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==-arr1).all()
    p2.set(1)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==zeros).all()
    print()

def test_02():
    outputs=[o.single() for o in [pp1, points1]]
    ws = WeightedSumP(outputs)

    ws.print()

    print( 'Mode1: a1*w1+a2*w2' )
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==arr1).all()
    p1.set(2)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==2.0*arr1).all()
    p2.set(2)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==2.0*arr1).all()
    p1.set(1)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==arr1).all()
    p2.set(1)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==arr1).all()
    print()

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')
