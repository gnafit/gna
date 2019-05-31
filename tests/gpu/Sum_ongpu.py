#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
import gna.constructors as C
from load import ROOT as R
from gna.constructors import Points, stdvector
from gna.env import env
from gna import context, bindings
import time

"""Initialize inpnuts"""
ndata = 3000
arr1 = N.arange(1, ndata)
arr2 = arr1
arr3 = arr1
print( 'Data1:', arr1 )
print( 'Data2:', arr2 )
print( 'Data3:', arr3 )

labels = [ 'arr1', 'arr2', 'arr3' ]
weights = [ 'w1', 'w2', 'w3' ]

"""Initialize environment"""

#with C.precision('double'):

with context.manager(ndata) as manager:
  """Initialize transformations"""
  points1 = Points( arr1 )
  points1.points.setLabel("T1")
#  points2 = Points( arr2 )
#  points2.points.setLabel("T2")
#  points3 = Points( arr3 )
#  points3.points.setLabel("T3")
    
  """Mode1: a1+a2"""
  ws = R.Sum()
  ws.add(points1.points)
#  ws.add(points2.points)
#  ws.add(points3.points)
  ws.sum.setLabel("T4")
    
#  ws.sum.switchFunction("gpu")
print( 'Mode1: ' )
print(  ws.sum.sum.data() )
print()


from gna.graphviz import GNADot

graph = GNADot( ws.sum )
graph.write("dotfile.dot")

   
for x in range(0,20):
    ws.sum.taint()
    start_time = time.time()
    ws.sum.sum.data()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    
