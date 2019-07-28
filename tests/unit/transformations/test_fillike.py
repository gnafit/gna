#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the FillLike transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna import constructors as C
from gna.env import env
from gna.unittest import *

def compare_filllike(a, b, message):
    if N.any(a!=b) and __name__!='__main__':
        raise Exception(message)

def test_filllike_v01(function_name='test_filllike_v01'):
    """Initialize inpnuts"""
    size = 5
    arr1 = N.arange(0, size)
    """Initialize environment"""
    ns=env.globalns(function_name)
    p1 = ns.defparameter( 'w1', central=1.0, sigma=0.1 )
    points1 = C.Points( arr1 )

    with ns:
        ws = C.WeightedSum(['w1'], [points1.points.points])
    ws.print()
    print()

    dbg1 = R.DebugTransformation('wsum')
    dbg1.debug.source(ws.sum.sum)

    flvalue = 2.0
    fl = C.FillLike(flvalue)
    fl.fill.inputs[0](dbg1.debug.target)

    dbg2 = R.DebugTransformation('fill')
    dbg2.debug.source(fl.fill.outputs[0])

    data=dbg2.debug.target.data()
    print('data:', data)
    print()
    compare_filllike(data, [flvalue]*size, 'Data output failed')

    print('Change parameter')
    p1.set(-1.0)
    taintflag = dbg2.debug.tainted()
    data=dbg2.debug.target.data()
    print('data:', data)
    print('taintflag:', taintflag)

    compare_filllike(data, [flvalue]*size, 'Data output failed')
    compare_filllike(taintflag, False, 'Taintflag should be false')

@floatcopy(globals(), addname=True)
def test_filllike_v02(function_name):
    """Initialize inpnuts"""
    size = 5
    arr1 = N.arange(0, size)
    """Initialize environment"""
    ns=env.globalns(function_name)
    p1 = ns.defparameter( 'w1', central=1.0, sigma=0.1 )
    points1 = C.Points( arr1 )

    with ns:
        ws = C.WeightedSum(['w1'], [points1.points.points])
    ws.print()
    print()

    flvalue = 2.0
    fl = C.FillLike(flvalue)
    ws >> fl.fill.inputs[0]
    out = fl.fill.outputs[0]

    data=out.data()
    print('data:', data)
    print()
    compare_filllike(data, [flvalue]*size, 'Data output failed')

    print('Change parameter')
    p1.set(-1.0)
    taintflag = fl.fill.tainted()
    print('data:', data)
    print('taintflag:', taintflag)

    compare_filllike(data, [flvalue]*size, 'Data output failed')
    compare_filllike(taintflag, False, 'Taintflag should be false')

if __name__ == "__main__":
    run_unittests(globals())
