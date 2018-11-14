#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.constructors import Points, stdvector
from gna.env import env

"""Initialize inpnuts"""
arr1 = N.arange(0, 5)
arr2 = -arr1
print( 'Data1:', arr1 )
print( 'Data2:', arr2 )

labels = [ 'arr1', 'arr2' ]
weights = [ 'w1', 'w2' ]

"""Initialize environment"""
p1 = env.globalns.defparameter( weights[0], central=1.0, sigma=0.1 )
p2 = env.globalns.defparameter( weights[1], central=1.0, sigma=0.1 )

"""Initialize transformations"""
points1 = Points( arr1 )
points2 = Points( arr2 )

"""Mode1: a1*w1+a2*w2"""
ws = R.WeightedSum( stdvector(labels), stdvector(weights) )
ws.sum.arr1(points1.points)
ws.sum.arr2(points2.points)

print( 'Mode1: a1*w1+a2*w2' )
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p2.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(1)
p2.set(1)
print()

"""Mode2: a1*w1+a2"""
ws = R.WeightedSum( stdvector(labels), stdvector(weights[:1]) )
ws.sum.arr1(points1.points)
ws.sum.arr2(points2.points)

print( 'Mode2: a1*w1+a2' )
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p2.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(1)
p2.set(1)
print()

"""Mode3: a1*w1+w2"""
ws = R.WeightedSum( stdvector(labels[:1]), stdvector(weights) )
ws.sum.arr1(points1.points)

print( 'Mode3: a1*w1+w2' )
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p2.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(1)
p2.set(1)
print()


"""Mode4: c+a1*w1+a1*w2"""
ws = R.WeightedSum( -10, stdvector(labels), stdvector(weights) )
ws.sum.arr1(points1.points)
ws.sum.arr2(points2.points)

print( 'Mode4: -10+a1*w1+a2*w2' )
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p2.set(2)
print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
p1.set(1)
p2.set(1)
print()
