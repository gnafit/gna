#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
from gna.unittest import run_unittests
from gna.env import env
import gna.constructors as C
import numpy as N

def test_varlist():
    ns = env.globalns('test_varlist')

    names  = [ 'zero', 'one', 'two', 'three', 'four', 'five' ]
    values = N.arange(len(names), dtype='d')
    for name, value in zip(names, values):
        ns.defparameter(name, central=value, relsigma=0.1)

    with ns:
        va = R.VarArray(C.stdvector(names))

    res=va.vararray.points.data()

    print('Python array:', values)
    print('Array:', res)

    # vlist = va.getFloatVariables()
    import IPython; IPython.embed()

if __name__ == '__main__':
    run_unittests(globals())
