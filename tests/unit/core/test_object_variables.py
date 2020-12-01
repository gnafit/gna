#!/usr/bin/env python

from load import ROOT as R
from gna.unittest import *
from gna.env import env
import gna.constructors as C
import numpy as N
from gna.bindings import parameters
from gna import context

@floatcopy(globals(), True)
def test_object_variables(function_name):
    ns = env.globalns(function_name)

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype=context.current_precision_short())
    for name, value in zip(names, values):
        ns.defparameter(name, central=value, relsigma=0.1)

    with ns:
        va = C.VarArray(names)

    for i, (name, val_true) in enumerate(zip(names, values)):
        var = va.variables[i].getVariable()
        assert name==var.name()

        val = var.cast().value()
        assert val==val_true

    for i, (name, val_true) in enumerate(zip(names, values)):
        ns[name].set(val_true)
        var=va.variables[i].getVariable()
        val=var.cast().value()
        assert val==val_true

if __name__ == '__main__':
    run_unittests(globals())
