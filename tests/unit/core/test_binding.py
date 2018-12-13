#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C

constant = 1.2345

def make(nsources, ntargets):
    sources=[C.Points([constant]) for i in xrange(nsources)]
    targets=[R.DebugTransformation('debug_%02d'%i) for i in xrange(ntargets)]

    return sources+targets

def check(*objs):
    for obj in objs:
        res = obj.transformations[0].outputs[0].data()[0]
        print('   ', res, 'should be', constant)
        assert res==constant

def test_single_01():
    dbg = R.DebugTransformation('debug')

    assert dbg.single()==dbg.transformations[0].outputs[0]
    assert dbg.single_input()==dbg.transformations[0].inputs[0]
    assert dbg.transformations[0].single()==dbg.transformations[0].outputs[0]
    assert dbg.transformations[0].single_input()==dbg.transformations[0].inputs[0]

def test_single_02():
    points, dbg = make(1, 1)
    points.points.points >> dbg.debug.source
    dbg.add_input()

    try:
        ret = dbg.single()
    except Exception:
        pass
    else:
        print(ret)
        assert False

def test_single_03():
    points, dbg = make(1, 1)
    points.points.points >> dbg.debug.source
    dbg.add_input()

    try:
        ret = dbg.single_input()
    except Exception:
        pass
    else:
        print(ret)
        assert False

def test_single_04():
    dbg = R.DebugTransformation('debug')
    dbg.add_transformation()

    try:
        ret = dbg.single()
    except Exception:
        pass
    else:
        print(ret)
        assert False

def test_single_05():
    dbg = R.DebugTransformation('debug')
    dbg.add_transformation()

    try:
        ret = dbg.single_input()
    except Exception:
        pass
    else:
        print(ret)
        assert False

def test_binding_01():
    points, dbg = make(1, 1)
    points.points.points >> dbg.debug.source
    check(dbg)

def test_binding_02():
    points, dbg = make(1, 1)
    dbg.debug.source << points.points.points
    print('    ', end='')
    check(dbg)

def test_binding_03():
    points, dbg1, dbg2 = make(1, 2)
    points.points.points >> dbg1.debug
    dbg2.debug << points.points.points
    check(dbg1, dbg2)

def test_binding_04():
    points, dbg1, dbg2 = make(1, 2)
    points.points.points >> dbg1
    dbg2 << points.points.points
    check(dbg1, dbg2)

def test_binding_05():
    points, dbg1, dbg2, dbg3 = make(1, 3)
    points.points.points >> (dbg1, dbg2, dbg3)
    check(dbg1, dbg2, dbg3)

def test_binding_06():
    points, dbg1, dbg2, dbg3 = make(1, 3)
    (dbg1, dbg2, dbg3) << points.points.points
    check(dbg1, dbg2, dbg3)

def test_binding_07():
    points, dbg1, dbg2 = make(1, 2)
    points.points >> dbg1.debug.source
    dbg2.debug.source << points.points
    check(dbg1, dbg2)

def test_binding_08():
    points, dbg1, dbg2 = make(1, 2)
    points >> dbg1.debug.source
    dbg2.debug.source << points
    check(dbg1, dbg2)

def test_binding_09():
    points, dbg1, dbg2, dbg3 = make(1, 3)
    points >> (dbg1, dbg2, dbg3)
    check(dbg1, dbg2, dbg3)

def test_binding_10():
    points, dbg1, dbg2, dbg3 = make(1, 3)
    (dbg1, dbg2, dbg3) << points
    check(dbg1, dbg2, dbg3)

def test_binding_11():
    points, dbg1, dbg2, dbg3 = make(1, 3)
    dbg1.add_transformation()
    try:
        (dbg1, dbg2, dbg3) << points
    except Exception:
        pass
    else:
        assert False

def test_binding_12():
    points1, points2, dbg1, dbg2, dbg3 = make(2, 3)
    points1>>dbg1
    dbg1.add_input()
    try:
        (dbg1, dbg2, dbg3) << points
    except Exception:
        pass
    else:
        assert False

if __name__ == "__main__":
    for fcn in sorted([fcn for name, fcn in globals().items() if name.startswith('test_')]):
        print('Call', fcn)
        fcn()
        print()

    print('All tests are OK!')
