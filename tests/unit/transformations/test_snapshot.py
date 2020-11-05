#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import env
from gna import context

@floatcopy(globals(), addname=True)
def test_snapshot_01(function_name):
    """Test ViewRear on Points (start, len)"""
    size=4
    ns = env.env.globalns(function_name)
    names=[]
    for i in range(size):
        name='val_%02i'%i
        names.append(name)
        ns.defparameter( name, central=i, fixed=True, label='Value %i'%i )

    with ns:
        vararray = C.VarArray(names)

    snapshot = C.Snapshot(vararray)

    for ichange, iname in enumerate(['']+names, -1):
        print('  Change', ichange)
        if iname:
            par=ns[iname]
            par.set(par.value()+1.0)

        res = snapshot.snapshot.result.data()
        vars = vararray.single().data()

        print('    Result', res)
        print('    Vars', vars)

        tainted = snapshot.snapshot.tainted()
        print('    Taintflag', tainted)
        assert not tainted

        if iname:
            assert (res!=vars).any()
        else:
            assert (res==vars).all()
        print()

    snapshot.snapshot.unfreeze()
    assert snapshot.snapshot.tainted()
    res = snapshot.snapshot.result.data()
    print('Result', res)
    print('Vars', vars)

    assert (res==vars).all()

if __name__ == "__main__":
    run_unittests(globals())

