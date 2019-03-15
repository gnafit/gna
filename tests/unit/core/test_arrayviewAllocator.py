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

def test_variable_allocation1():
    from gna.env import env
    gns = env.globalns('test_variable_allocation1')
    ndata = 10
    allocator = allocator_d(ndata)

    with context.allocator(allocator):
        ns = gns('namespace1')
        ns.defparameter('float1',   central=1, fixed=True, label='Float variable 1')
        ns.defparameter('float2',   central=2, fixed=True, label='Float variable 2')
        ns = gns('namespace2')
        ns.defparameter('angle',    central=3, label='Angle parameter', type='uniformangle')
        ns.defparameter('discrete', default='a', label='Discrete parameter', type='discrete', variants=OrderedDict([('a', 10.0), ('b', 20.0), ('c', 30.0)]))

    ns = gns('namespace3')
    ns.defparameter('float3',   central=100, fixed=True, label='Float variable 3 (independent)')
    ns.defparameter('float4',   central=200, fixed=True, label='Float variable 4 (independent)')

    gns.printparameters(labels=True)

    print('Data (filled):', allocator.view())
    print('Data (all):', allocator.viewall())

def test_variable_allocation_complex():
    from gna.env import env
    gns = env.globalns('test_variable_allocation_complex')
    ndata = 200
    allocator = allocator_d(ndata)

    with context.allocator(allocator):
        ns = gns('namespace1')
        ns.defparameter('float1',   central=1, fixed=True, label='Float variable 1')
        ns.defparameter('float2',   central=2, fixed=True, label='Float variable 2')

        ns = gns('namespace2')
        ns.defparameter('angle',    central=3, label='Angle parameter', type='uniformangle')
        ns.defparameter('discrete', default='a', label='Discrete parameter', type='discrete', variants=OrderedDict([('a', 10.0), ('b', 20.0), ('c', 30.0)]))

        ns = gns('namespace3')
        from gna.parameters.oscillation import reqparameters
        reqparameters(ns)
        with ns:
            ns.materializeexpressions()
        ns['Delta'].set(N.pi*1.5)

    gns.printparameters(labels=True)

    allocdata = allocator.view()
    print('Data (filled):', allocdata)
    print('Data (all):', allocator.viewall())

    for name, par in gns.walknames():
        av = par.getVariable().values()
        offset = av.offset()
        size = av.size()

        avview = av.view()
        assert (allocdata[offset:offset+size]==avview).all()

from gna.bindings import provided_precisions
if 'float' in provided_precisions:
    def test_variables_precision():
        from gna.env import env
        gns = env.globalns('test_variable_precision')

        ns = gns('namespace1')
        ns.defparameter('double',   central=1, fixed=True, label='Double precision variable')
        with context.precision('float'):
            ns.defparameter('float',   central=2, fixed=True, label='Float variable')

        var_dbl = ns['double'].getVariable()
        var_flt = ns['float'].getVariable()

        assert var_dbl.value()==1.0
        assert var_dbl.typeName()=='double'

        assert var_flt.value()==2.0
        assert var_flt.typeName()=='float'

        par_flt = ns['float'].getParameter()
        par_dbl = ns['double'].getParameter()
        par_flt.set(1.e-46)
        par_dbl.set(1.e-46)
        assert par_flt.value()==0.0
        assert par_dbl.value()!=0.0

        par_flt.set(1.e39)
        par_dbl.set(1.e39)
        assert par_flt.value()==float('inf')
        assert par_dbl.value()!=float('inf')

if __name__ == "__main__":
    run_unittests(globals())

