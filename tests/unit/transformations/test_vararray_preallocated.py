#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna.env import env
import gna.constructors as C
import numpy as N
from gna import context
import gna.bindings.arrayview

# @floatcopy(globals(), True)
@passname
def test_vararray_v01(function_name):
    ns = env.globalns(function_name)

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype=context.current_precision_short())
    variables = R.vector('variable<%s>'%context.current_precision())()

    with context.allocator(100) as allocator:
        for name, value in zip(names, values):
            par = ns.defparameter(name, central=value, relsigma=0.1)
            variables.push_back(par.getVariable())

        with ns:
            vsum = C.VarSum(names, 'sum', ns=ns)
            vsum_var=ns['sum'].get()
            variables.push_back(vsum_var.getVariable())
            vprod = C.VarProduct(names, 'product', ns=ns)
            vprod_var=ns['product'].get()
            variables.push_back(vprod_var.getVariable())

        va = C.VarArrayPreallocated(variables)

    pool=allocator.view()
    res=va.vararray.points.data()

    values_all = N.zeros(shape=values.size+2, dtype=values.dtype)
    values_all[:-2]=values
    values_all[-2]=values_all[:-2].sum()
    values_all[-1]=values_all[:-2].prod()

    print('Python array:', values_all)
    print('VarArray (preallocated):', res)
    print('Pool:', pool)

    assert (values_all==res).all()
    assert (values_all==pool).all()
    assert (res==pool).all()

    for i, (val, name) in enumerate(enumerate(names, 2)):
        ns[name].set(val)
        values_all[i]=val
        values_all[-2]=values_all[:-2].sum()
        values_all[-1]=values_all[:-2].prod()
        res=va.vararray.points.data()

        print('Iteration', i)
        print('    Python array:', values_all)
        print('    VarArray (preallocated):', res)

        assert (values_all==res).all()
        assert (values_all==pool).all()
        assert (res==pool).all()

if __name__ == '__main__':
    run_unittests(globals())
