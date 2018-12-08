#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from constructors import Points, WeightedSum
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

env.globalns.printparameters()

"""Initialize transformations"""
points1 = Points( arr1 )
points2 = Points( arr2 )

"""Mode1: a1*w1+a2*w2"""
ws = WeightedSum(weights, [points1.points.points, points2.points.points])

dbg1 = R.DebugTransformation('wsum')
dbg1.debug.source(ws.sum.sum)

fl = R.FillLike(2.0)
fl.fill.inputs[0](dbg1.debug.target)

dbg2 = R.DebugTransformation('fill')
dbg2.debug.source(fl.fill.outputs[0])

print(dbg2.debug.target.data())
print()

print('Change parameter')
p1.set(2.0)
print(dbg2.debug.target.data())
