#!/usr/bin/env python

from gna.unittest import *
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna import context, bindings

import gna.bindings.arrayview

# @floatcopy(globals(), True)
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
        ns.defparameter('discrete', default='a', label='Discrete parameter', type='discrete', variants=dict([('a', 10.0), ('b', 20.0), ('c', 30.0)]))

        ns = gns('namespace3')
        from gna.parameters.oscillation import reqparameters
        reqparameters(ns)
        with ns:
            ns.materializeexpressions()
        ns['Delta'].set(N.pi*0.25)

        pars = tuple(par.getVariable() for (name,par) in ns.walknames())
        manager.setVariables(C.stdvector(pars))

    gns.printparameters(labels=True)

    allocator = manager.getAllocator()
    varray = manager.getVarArray()
    data_v = varray.vararray.points.data()
    data_a = allocator.view()
    print('Data (filled):', data_a.size, data_a.dtype, data_a)
    print('Data (vararray):', data_v.size, data_v.dtype, data_v)
    mask = data_v==data_a
    assert mask is not False and mask.all()

if __name__ == "__main__":
    run_unittests(globals())

