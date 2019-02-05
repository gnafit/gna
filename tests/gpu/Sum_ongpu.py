#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
import gna.constructors as C
from load import ROOT as R
from gna.constructors import Points, stdvector
from gna.env import env


"""Initialize inpnuts"""
arr1 = N.arange(1, 1000000)
arr2 = arr1
arr3 = arr1
print( 'Data1:', arr1 )
print( 'Data2:', arr2 )
print( 'Data3:', arr3 )

labels = [ 'arr1', 'arr2', 'arr3' ]
weights = [ 'w1', 'w2', 'w3' ]

"""Initialize environment"""

with C.precision('double'):
    """Initialize transformations"""
    points1 = Points( arr1 )
    points2 = Points( arr2 )
    points3 = Points( arr3 )
    
    """Mode1: a1+a2"""
    ws = R.Sum()
    ws.add(points1.points)
    ws.add(points2.points)
    ws.add(points3.points)
    
    ws.sum.switchFunction("gpu")
    print( 'Mode1: ' )
    print(  ws.sum.sum.data() )
    print()
    
