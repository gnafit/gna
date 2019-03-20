#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import *
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna import context, bindings
from collections import OrderedDict
import gna.bindings.arrayview

#@floatcopy(globals(), True)
@passname
def test_tree_manager(function_name):
    from gna.env import env
    gns = env.globalns(function_name)
    ndata = 200

    with context.manager(ndata) as manager:
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
        ns['Delta'].set(N.pi*0.25)

        with gns:
            names=[]
            names_all=[]
            indices=[]
            for name, par in gns.walknames():
                names.append(name)
                val=par.getVariable().values()
                offset = val.offset()
                # if val.size()>1:
                    # import IPython; IPython.embed()
                for i in range(val.size()):
                    indices.append(offset+i)
                    names_all.append(name)
            varray = C.VarArray(names)

    gns.printparameters(labels=True)

    allocator = manager.getAllocator()
    data_v = varray.vararray.points.data()
    data_a = allocator.view()
    data_s = data_a[indices].copy()
    print('Data (filled):', data_a.size, data_a.dtype, data_a)
    print('Data (vararray):', data_v.size, data_v.dtype, data_v)
    print('Data (selected):', data_s.size, data_s.dtype, data_s)
    mask = data_v==data_s
    assert mask is not False and mask.all()

if __name__ == "__main__":
    run_unittests(globals())

