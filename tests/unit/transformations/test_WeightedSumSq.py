#!/usr/bin/env python

"""Check the WeightedSumSq transformation"""

from gna.unittest import *
import numpy as N
from load import ROOT as R
from gna import constructors as C
from gna.env import env

def weightedsumsq_make(nsname):
    """Initialize inpnuts"""
    arr1 = N.arange(0, 5, dtype='d')
    arr2 = -arr1
    zeros = N.zeros((5,), dtype='d')
    print( 'Data1:', arr1 )
    print( 'Data2:', arr2 )

    labels = [ 'arr1', 'arr2' ]
    weights = [ 'w1', 'w2' ]

    """Initialize environment"""
    ns=env.globalns(nsname)
    p1 = ns.defparameter( weights[0], central=1.0, sigma=0.1 )
    p2 = ns.defparameter( weights[1], central=1.0, sigma=0.1 )

    ns.printparameters()

    """Initialize transformations"""
    points1 = C.Points( arr1 )
    points2 = C.Points( arr2 )

    return ns, weights, arr1, p1, points1, arr2, p2, points2, zeros

@floatcopy(globals(), addname=True)
def test_weightedsumsq_01(function_name):
    ns, weights, arr1, p1, points1, arr2, p2, points2, zeros = weightedsumsq_make(function_name)
    outputs=[o.single() for o in [points1, points2]]
    with ns:
        ws = C.WeightedSumSq(weights, outputs)

    ws.print()

    print( 'Mode1: a1*w1+a2*w2' )
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==(arr1**2+arr2**2)).all()
    p1.set(2)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==(arr1**2+4*arr2**2)).all()
    p2.set(2)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==(4*arr1**2+4*arr2**2)).all()
    p1.set(1)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==(arr1**2+4*arr2**2)).all()
    p2.set(1)
    print( '  ', p1.value(), p2.value(), ws.sum.sum.data() )
    assert (ws.sum.sum.data()==(arr1**2+arr2**2)).all()
    print()


if __name__ == "__main__":
    run_unittests(globals())
