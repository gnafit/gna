#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from gna import constructors as C
import numpy as np

def test_varsum():
    ns = env.globalns
    names = [ 'one', 'two', 'three', 'four', 'five' ]
    for i, name in enumerate( names, 1 ):
        ns.defparameter( name, central=i, relsigma=0.1 )

    vp = C.VarSum(names, 'sum', ns=ns)
    ns['sum'].get()

    summed_cpp = ns['sum'].value()
    summed_python =  sum(range(1, len(names)+1))
    print_parameters(ns)
    assert(np.allclose(summed_cpp, summed_cpp))

    print('Change one input at a time:')
    for i, name in enumerate( names, 2 ):
        ns[name].set(i)
        print_parameters(ns)
        print()
        summed_python += 1 
        summed_cpp = ns['sum'].value()
        assert(np.allclose(summed_cpp, summed_cpp))

if __name__ == "__main__":
    test_varsum()

