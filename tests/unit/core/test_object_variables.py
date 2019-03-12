#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
from gna.unittest import run_unittests
from gna.env import env
import gna.constructors as C
import numpy as N
from gna.bindings import parameters

def test_object_variables():
    ns = env.globalns('test_varlist')

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype='d')
    for name, value in zip(names, values):
        ns.defparameter(name, central=value, relsigma=0.1)

    with ns:
        va = R.VarArray(C.stdvector(names))

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
