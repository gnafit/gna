#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from constructors import Points, stdvector
from gna.env import env

"""Initialize inpnuts"""
arr1 = N.arange(0, 5)
arr2 = arr1
print( 'Data1:', arr1 )
print( 'Data2:', arr2 )

labels = [ 'arr1', 'arr2' ]
weights = [ 'w1', 'w2' ]

"""Initialize environment"""

"""Initialize transformations"""
points1 = Points( arr1 )
points2 = Points( arr2 )

"""Mode1: a1*w1+a2*w2"""
ws = R.Sum()
ws.add(points1.points)
ws.add(points2.points)

print( 'Mode1: ' )
print(  ws.sum.sum.data() )
print()

