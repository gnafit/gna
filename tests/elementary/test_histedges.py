#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
import constructors as C

edges = N.linspace(0.0, 10.0, 11)
data  = N.arange(10.0)

hist = C.Histogram( edges, data )
h2e = R.HistEdges()
h2e.histedges.hist( hist.hist.hist )

e = h2e.histedges.edges.data()

print( 'Input:' )
print( edges )

print( 'Output:' )
print( e )

print( (edges==e).all() and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )

h2e1 = R.HistEdges()
print('\033[32mPlanned exception (wrong type):\033[0m')
h2e1.histedges.hist( h2e.histedges.edges )
