#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.constructors import Points, WeightedSum
from gna.env import env

def compare(a, b, message):
    if N.any(a!=b) and __name__!='__main__':
        raise Exception(message)

def test_filllike():
    """Initialize inpnuts"""
    size = 5
    arr1 = N.arange(0, size)
    """Initialize environment"""
    ns=env.globalns('test_filllike')
    p1 = ns.defparameter( 'w1', central=1.0, sigma=0.1 )
    points1 = Points( arr1 )

    with ns:
        ws = WeightedSum(['w1'], [points1.points.points])
    ws.print()
    print()

    dbg1 = R.DebugTransformation('wsum')
    dbg1.debug.source(ws.sum.sum)

    flvalue = 2.0
    fl = R.FillLike(flvalue)
    fl.fill.inputs[0](dbg1.debug.target)

    dbg2 = R.DebugTransformation('fill')
    dbg2.debug.source(fl.fill.outputs[0])

    data=dbg2.debug.target.data()
    print('data:', data)
    print()
    compare(data, [flvalue]*size, 'Data output failed')

    print('Change parameter')
    p1.set(-1.0)
    taintflag = dbg2.debug.tainted()
    data=dbg2.debug.target.data()
    print('data:', data)
    print('taintflag:', taintflag)

    compare(data, [flvalue]*size, 'Data output failed')
    compare(taintflag, False, 'Taintflag should be false')

if __name__ == "__main__":
    test_filllike()
