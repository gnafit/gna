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

def check(obj):
    res = obj.transformations[0].outputs[0].data()
    print(res, 'should be', constant)
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

    try:
        ret = dbg.single_input()
    except Exception:
        pass
    else:
        print(ret)
        assert False

def test_single_03():
    dbg = R.DebugTransformation('debug')
    dbg.add_transformation()

    dbg.print()
    import IPython
    IPython.embed()

    try:
        ret = dbg.single()
    except Exception:
        pass
    else:
        print(ret)
        assert False

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
    check(dbg)

# def test_binding_03():
    # points1, points2, dbg1, dbg2 = make(2, 2)
    # dbg1.debug << points.points.points
    # check(dbg)


if __name__ == "__main__":
    for fcn in sorted([fcn for name, fcn in globals().items() if name.startswith('test_')]):
        print('Call', fcn)
        fcn()
        print()

    print('All tests are OK!')
