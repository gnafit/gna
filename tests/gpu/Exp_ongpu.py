#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
import gna.constructors as C
from load import ROOT as R
from gna.constructors import Points, Exp, stdvector
from gna.env import env
from gna import context, bindings
import time

"""Initialize inpnuts"""
ndata =1300
arr1 = N.arange(0.001, ndata/100, 0.001)
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
  points2 = Points( arr2 )
  points3 = Points( arr3 )
    
  """Mode1: a1+a2"""
  ws = Exp()
  ws.exp.points(points1.points)
    
  ws.exp.switchFunction("gpu")
print( 'Mode1: ' )
print(  ws.exp.result.data() )
print()

   
N=1
start_time = time.time()
for x in range(N):
    ws.exp.taint()

end_time = time.time()
fake_time = end_time - start_time
print('Fake time', fake_time)

start_time = time.time()
for x in range(N):
    ws.exp.taint()
    ws.exp.result.data()
end_time = time.time()
elapsed_time = end_time - start_time
print('Total time', elapsed_time)

print('GNA time (%i trials)'%N, elapsed_time-fake_time)
print('GNA time per event', (elapsed_time-fake_time)/N)

