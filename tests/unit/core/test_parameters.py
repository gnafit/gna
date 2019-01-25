#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R

def check(text, before, after, shouldbe, taintflag):
    if before is not None:
        print('   ', text, 'before', before)

    print('   ', text, 'after', after, '=', shouldbe)
    print('   ', 'tainted', bool(taintflag))
    assert after==shouldbe

    if taintflag is None:
        print()
        return

    assert bool(taintflag)
    taintflag.set(False)
    assert not bool(taintflag)
    print()

def test_var_01():
    """Test getters"""
    print("Test getters (01)")
    var = R.parameter('double')('testpar')
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)

    const = 1.5
    print('Set', const)
    var.set(const)
    print('Tatinflag', bool(taintflag))
    taintflag.set(False)
    print('Tatinflag', bool(taintflag))

    check('ret scalar', None, var.value(), const, None)
    check('ret index[0]', None, var.value(0), const, None)
    check('ret vector', None, list(var.values()), [const], None)

    ret = N.zeros(1, dtype='d')
    before=ret.copy()
    var.values(ret)
    check('arg C array', before, ret, [const], None)

    ret = R.vector('double')(1)
    before=list(ret)
    var.values(ret)
    check('arg std vector', before, list(ret), [const], None)

def test_var_02():
    """Test setters"""
    print("Test setters (02)")
    var = R.parameter('double')('testpar')
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    const = 1.5
    var.set(const)
    const+=1.0

    check('scalar', None, var.value(), const, None)

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

