#!/usr/bin/env python

from load import ROOT as R
from gna.unittest import *
from gna.env import env
import gna.constructors as C
import numpy as N
from gna import context

def test_jacobian_v01():
    ns = env.globalns("test_jacobian_v01")

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype=context.current_precision_short())
    jac = C.Jacobian()
    for name, value in zip(names, values):
        if value:
            par = ns.defparameter(name, central=value, relsigma=0.1)
        else:
            par = ns.defparameter(name, central=value, sigma=0.1)

        if name=='five':
            break

        jac.append(par)
    five = par

    with ns:
        va = C.VarArray(names)

    va.vararray.points >> jac.jacobian.func

    vars=va.vararray.points.data()

    print('Python array:', values.shape, values)
    print('Array:', vars.shape, vars)

    res=jac.jacobian.jacobian.data()
    print('Jacobian:', res.shape, res)

    req = N.eye(len(names), len(names)-1)
    assert N.allclose(res, req, atol=1.e-12)

    t = jac.jacobian
    tf1 = t.getTaintflag()
    tf0 = va.vararray.getTaintflag()

    assert not tf0.tainted()
    assert not tf1.tainted()
    assert tf1.frozen()

    five.set(12.0)
    assert tf0.tainted()
    assert not tf1.tainted()
    assert tf1.frozen()

    t.unfreeze()
    assert tf0.tainted()
    assert tf1.tainted()
    assert not tf1.frozen()

    res=jac.jacobian.jacobian.data()
    assert N.allclose(res, req, atol=1.e-12)
    assert not tf0.tainted()
    assert not tf1.tainted()
    assert tf1.frozen()

if __name__ == '__main__':
    run_unittests(globals())
