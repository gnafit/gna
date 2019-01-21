#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Demonstrates the replace() function operation of the variable instance"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
R.GNAObject

def test_gauss_par_repl():
    p1 = R.GaussianParameter('double')('test1')
    p1.set(-1.0)

    p2 = R.GaussianParameter('double')('test2')
    p2.set(-2.0)

    def prt( title ):
        if title:
            print( title )
        print( '  p1', p1.name(), p1.value() )
        print( '  p2', p2.name(), p2.value() )
        print()

    prt( 'Before replacement' )

    p2.getVariable().replace( p1.getVariable() )
    #  p2.getParameter().replace( p1.getVariable() )
    prt( 'After replacement' )
    assert(p1.value() == p2.value())

    p1.set(1)
    prt( 'Change parameter 1' )
    assert(p1.value() == p2.value())

    p2.set(2)
    prt( 'Change parameter 2' )
    assert(p1.value() == p2.value())

if __name__ == "__main__":
    test_gauss_par_repl()
