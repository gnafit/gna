#!/usr/bin/env python

from load import ROOT as R
from gna.unittest import *
from gna.env import env
import gna.constructors as C
import numpy as N
from gna import context

@floatcopy(globals(), True)
def test_vararray_v01(function_name):
    ns = env.globalns(function_name)

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype=context.current_precision_short())
    for name, value in zip(names, values):
        ns.defparameter(name, central=value, relsigma=0.1)

    with ns:
        va = C.VarArray(names)

    res=va.vararray.points.data()

    print('Python array:', values)
    print('Array:', res)

    assert N.allclose(values, res, atol=0, rtol=0)

    for i, (val, name) in enumerate(enumerate(names, 2)):
        ns[name].set(val)
        values[i]=val
        res=va.vararray.points.data()
        assert N.allclose(values, res, atol=0, rtol=0)

@floatcopy(globals(), True)
def test_vararray_v02(function_name):
    ns = env.globalns(function_name)

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype=context.current_precision_short())
    variables = R.vector('variable<%s>'%context.current_precision())()
    for name, value in zip(names, values):
        par = ns.defparameter(name, central=value, relsigma=0.1)
        variables.push_back(par.getVariable())

    va = C.VarArray(variables)

    res=va.vararray.points.data()

    print('Python array:', values)
    print('Array:', res)

    assert N.allclose(values, res, atol=0, rtol=0)

    for i, (val, name) in enumerate(enumerate(names, 2)):
        ns[name].set(val)
        values[i]=val
        res=va.vararray.points.data()
        assert N.allclose(values, res, atol=0, rtol=0)

if __name__ == '__main__':
    run_unittests(globals())
