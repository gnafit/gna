#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna import constructors as C

def check(text, before, after, shouldbe, taintflag, tainted=True):
    if before is not None:
        print('   ', text, 'before', before)

        assert N.all(before!=shouldbe)

    print('   ', text, 'after', after, '=', shouldbe)
    print('   ', 'tainted', bool(taintflag))
    assert N.all(after==shouldbe)

    if taintflag is None:
        print()
        return

    assert bool(taintflag)==tainted
    taintflag.set(False)
    assert not bool(taintflag)
    print()

def test_var_01():
    """Test getters"""
    print("Test getters")
    var = R.parameter('double')('testpar')
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)

    const = 1.5
    print('Set', const)
    var.set(const)
    print('Taintflag', bool(taintflag))

    check('ret scalar', None, var.value(), const, taintflag, True)
    check('ret index[0]', None, var.value(0), const, taintflag, False)
    check('ret vector', None, list(var.values()), [const], taintflag, False)

    ret = N.zeros(1, dtype='d')
    before=ret.copy()
    var.values(ret)
    check('arg C array', before, ret, [const], taintflag, False)

    ret = R.vector('double')(1)
    before=list(ret)
    var.values(ret)
    check('arg std vector', before, list(ret), [const], taintflag, False)

def test_var_02():
    """Test getters (vec)"""
    print("Test getters")
    const = N.array([1.5, 2.6, 3.7], dtype='d')

    var = R.parameter('double')('testpar', const.size)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)

    print('Set', const)
    var.set(const)
    print('Taintflag', bool(taintflag))

    check('ret scalar', None, var.value(), const[0], taintflag, True)
    for i, val in enumerate(const):
        check('ret index[%i]'%i, None, var.value(i), val, taintflag, False)
    check('ret vector', None, list(var.values()), const, taintflag, False)

    ret = N.zeros(const.size, dtype='d')
    before=ret.copy()
    var.values(ret)
    check('arg C array', before, ret, const, taintflag, False)

    ret = R.vector('double')(const.size)
    before=list(ret)
    var.values(ret)
    check('arg std vector', before, list(ret), const, taintflag, False)

def test_var_03():
    """Test setters"""
    print("Test setters")
    var = R.parameter('double')('testpar')
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    const = 1.5
    var.set(const)
    check('scalar', None, var.value(), const, taintflag)

    const+=1.0
    var.set(0, const)
    check('index [0]', None, var.value(), const, taintflag)

    const+=1.0
    arr = N.array([const], dtype='d')
    var.set(arr)
    check('C array', None, var.value(), const, taintflag)

    const+=1.0
    arr = R.vector('double')(1, const)
    var.set(arr)
    check('std vector', None, var.value(), const, taintflag)

def test_var_04():
    """Test setters"""
    print("Test setters")
    const = N.array([1.5, 2.6, 3.7], dtype='d')
    var = R.parameter('double')('testpar', const.size)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    var.set(const)
    check('C array', None, list(var.values()), const, taintflag)

    const+=1.0
    vec = C.stdvector(const)
    var.set(vec)
    check('std vector', None, list(var.values()), const, taintflag)

def test_var_05():
    """Test nested vector"""
    print("Test setters")
    const = N.array([1.5, 2.6, 3.7], dtype='d')
    var = R.parameter('vector<double>')('testpar')
    var.value().resize(3)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    vec = C.stdvector(const)
    var.set(vec)

    check('vec', None, list(var.value()), const, taintflag)

def test_var_06():
    """Test nested vector"""
    print("Test setters")
    const = N.array([1.5, 2.6, 3.7], dtype='d')
    var = R.parameter('vector<double>')('testpar',2)
    var.value(0).resize(3)
    var.value(1).resize(3)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    vec = C.stdvector(const)
    var.set(0, vec)
    vec = C.stdvector(const+1.0)
    var.set(1, vec)

    check('vec 0', None, list(var.value(0)), const, taintflag)
    check('vec 1', None, list(var.value(1)), const+1.0, taintflag, False)

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

