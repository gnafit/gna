#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import run_unittests, makefloat
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna import context
from collections import OrderedDict

def arrayview_d_view(self):
    return N.frombuffer(self.data(), count=self.size(), dtype='d')

def allocator_d_viewall(self):
    return N.frombuffer(self.data(), count=self.maxSize(), dtype='d')

arrayview_d = R.arrayview('double')
arrayview_d.view = arrayview_d_view
allocator_d = R.arrayviewAllocatorSimple('double')
allocator_d.view = arrayview_d_view
allocator_d.viewall = allocator_d_viewall

def test_arrayview_allocation():
    nitems, ndata = 6, 15
    allocator = allocator_d(ndata)

    arrays = []
    for i in range(1, nitems):
        array = arrayview_d(i, allocator)
        for j in range(i):
            array[j]=j
        print(i, array.view())
        arrays.append(array)

    print('Arrays:', [array.view() for array in arrays])
    print('Data (filled):', allocator.view())
    print('Data (all):', allocator.viewall())

def test_variable_allocation():
    from gna.env import env
    ns = env.globalns('test_variable_allocation')
    ndata = 10
    allocator = allocator_d(ndata)

    with context.allocator(allocator):
        ns.defparameter('float1',   central=1, fixed=True, label='Float variable 1')
        ns.defparameter('float2',   central=2, fixed=True, label='Float variable 2')
        ns.defparameter('angle',    central=3, label='Angle parameter', type='uniformangle')
        ns.defparameter('discrete', default='a', label='Discrete parameter', type='discrete', variants=OrderedDict([('a', 10.0), ('b', 20.0), ('c', 30.0)]))

    ns.printparameters(labels=True)

    print('Data (filled):', allocator.view())
    print('Data (all):', allocator.viewall())

if __name__ == "__main__":
    run_unittests(globals())

