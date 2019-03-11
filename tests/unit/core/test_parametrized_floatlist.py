#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT
from gna.unittest import run_unittests
from gna.env import env
import gna.constructors as C

def test_varlist():
    ns = env.globalns('test_varlist')

    names = [ 'one', 'two', 'three', 'four', 'five' ]
    for i, name in enumerate( names ):
        ns.defparameter( name, central=i, relsigma=0.1 )

    vnames = C.stdvector( names )
    with ns:
        va = R.VarArray(vnames)

    ns.printparameters()
    print( 'Array:', output.data() )

if __name__ == '__main__':
    run_unittests(globals())
