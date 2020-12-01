#!/usr/bin/env python

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

def test_par_01(floatprecision='double'):
    """Test getters"""
    assert floatprecision in ['double', 'float']
    var = R.parameter(floatprecision)('testpar')
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)

    const = 1.5
    print('Set', const)
    var.set(const)
    print('Taintflag', bool(taintflag))

    check('ret scalar', None, var.value(), const, taintflag, True)
    check('ret index[0]', None, var.value(0), const, taintflag, False)
    check('ret vector', None, list(var.values()), [const], taintflag, False)

    ret = N.zeros(1, dtype=floatprecision[0])
    before=ret.copy()
    var.values(ret)
    check('arg C array', before, ret, [const], taintflag, False)

    ret = R.vector(floatprecision)(1)
    before=list(ret)
    var.values(ret)
    check('arg std vector', before, list(ret), [const], taintflag, False)

def test_par_02(floatprecision='double'):
    """Test getters (vec)"""
    assert floatprecision in ['double', 'float']
    const = N.array([1.5, 2.6, 3.7], dtype=floatprecision[0])

    var = R.parameter(floatprecision)('testpar', const.size)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)

    print('Set', const)
    var.set(const)
    print('Taintflag', bool(taintflag))

    check('ret scalar', None, var.value(), const[0], taintflag, True)
    for i, val in enumerate(const):
        check('ret index[%i]'%i, None, var.value(i), val, taintflag, False)
    check('ret vector', None, list(var.values()), const, taintflag, False)

    ret = N.zeros(const.size, dtype=floatprecision[0])
    before=ret.copy()
    var.values(ret)
    check('arg C array', before, ret, const, taintflag, False)

    ret = R.vector(floatprecision)(const.size)
    before=list(ret)
    var.values(ret)
    check('arg std vector', before, list(ret), const, taintflag, False)

def test_par_03(floatprecision='double'):
    """Test setters"""
    assert floatprecision in ['double', 'float']
    var = R.parameter(floatprecision)('testpar')
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
    arr = N.array([const], dtype=floatprecision[0])
    var.set(arr)
    check('C array', None, var.value(), const, taintflag)

    const+=1.0
    arr = R.vector(floatprecision)(1, const)
    var.set(arr)
    check('std vector', None, var.value(), const, taintflag)

def test_par_04(floatprecision='double'):
    """Test setters"""
    assert floatprecision in ['double', 'float']
    const = N.array([1.5, 2.6, 3.7], dtype=floatprecision[0])
    var = R.parameter(floatprecision)('testpar', const.size)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    var.set(const)
    check('C array', None, list(var.values()), const, taintflag)

    const+=1.0
    vec = C.stdvector(const)
    var.set(vec)
    check('std vector', None, list(var.values()), const, taintflag)

def test_par_05(floatprecision='double'):
    """Test nested vector"""
    assert floatprecision in ['double', 'float']
    const = N.array([1.5, 2.6, 3.7], dtype=floatprecision[0])
    var = R.parameter('vector<%s>'%floatprecision)('testpar')
    var.value().resize(3)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    vec = C.stdvector(const)
    var.set(vec)

    check('vec', None, list(var.value()), const, taintflag)

def test_par_06(floatprecision='double'):
    """Test nested vector"""
    assert floatprecision in ['double', 'float']
    const = N.array([1.5, 2.6, 3.7], dtype=floatprecision[0])
    var = R.parameter('vector<%s>'%floatprecision)('testpar',2)
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

def test_var_01(floatprecision='double'):
    """Test variable"""
    assert floatprecision in ['double', 'float']
    const = N.array([1.5, 2.6, 3.7], dtype=floatprecision[0])
    par = R.parameter('vector<%s>'%floatprecision)('testpar')
    par.value().resize(3)

    var = R.variable('vector<%s>'%floatprecision)(par)
    taintflag = R.taintflag('tflag')
    var.subscribe(taintflag)
    taintflag.set(False)

    vec = C.stdvector(const)
    par.set(vec)
    check('vec', None, list(var.value()), const, taintflag)

def test_evaluable_01(floatprecision='double'):
    """Test evaluable"""
    assert floatprecision in ['double']

    taintflag = R.taintflag('tflag')
    eva = R.GNAUnitTest.make_test_dependant_double('dep', 1)
    taintflag.subscribe(eva)

    for i in range(1, 10):
        check('iteration '+str(i), None, eva.value(), i, taintflag, True)
        check('iteration '+str(i), None, eva.value(), i, taintflag, False)
        check('iteration '+str(i), None, eva.value(), i, taintflag, False)
        taintflag.taint()

def test_evaluable_02(floatprecision='double'):
    """Test evaluable"""
    assert floatprecision in ['double']

    const = N.arange(1, 6, dtype=floatprecision[0])
    taintflag = R.taintflag('tflag')
    eva = R.GNAUnitTest.make_test_dependant_double('dep', const.size)
    taintflag.subscribe(eva)

    for i in range(1, 10):
        check('iteration '+str(i), None, list(eva.values()), const, taintflag, True)
        check('iteration '+str(i), None, list(eva.values()), const, taintflag, False)
        check('iteration '+str(i), None, list(eva.values()), const, taintflag, False)
        const+=const.size
        taintflag.taint()

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        fcn = glb[fcn]
        print(fcn.__doc__)
        fcn()
        print()

    print('All tests are OK!')

