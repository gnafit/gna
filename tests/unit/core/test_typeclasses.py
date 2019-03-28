#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
import numpy as np

def check(r, size, expect):
    what = list(r.vector(size, False))
    print('Check:', what, expect)
    assert what==expect

def test_range_01():
    r = R.TypeClasses.Range(0)
    r.dump(); print()

    check(r, 1, [0])
    check(r, 2, [0])
    check(r, 3, [0])

def test_range_02():
    r = R.TypeClasses.Range(1)
    r.dump(); print()

    check(r, 1, [])
    check(r, 2, [1])
    check(r, 3, [1])

def test_range_03():
    r = R.TypeClasses.Range(0,2)
    r.dump(); print()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [0,1,2])

def test_range_04():
    r = R.TypeClasses.Range(0,-1)
    r.dump(); print()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [0,1,2,3])

def test_range_05():
    r = R.TypeClasses.Range(2,-1)
    r.dump(); print()

    check(r, 1, [])
    check(r, 2, [])
    check(r, 3, [2])
    check(r, 4, [2,3])
    check(r, 5, [2,3,4])
    check(r, 6, [2,3,4,5])

def test_range_06():
    r = R.TypeClasses.Range(-3,-1)
    r.dump(); print()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [1,2,3])
    check(r, 5, [2,3,4])

def test_range_07():
    r = R.TypeClasses.Range(-3,2)
    r.dump(); print()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [1,2])
    check(r, 5, [2])
    check(r, 6, [])

@floatcopy(globals)
def test_typeclass_same_v01():
    """All inputs have same types"""
    arrays = [np.arange(12, dtype='d').reshape(3,4) for i in range(5)]
    points = [C.Points(a) for a in arrays]
    outputs = [p.points.points for p in points]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(C._current_precision)((1,-1))
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

@floatcopy(globals)
def test_typeclass_same_v02():
    """Last input has another shape"""
    arrays = [np.arange(12, dtype='d').reshape(3,4) for i in range(5)]
    arrays[-1]=arrays[-1].reshape(4,3)
    points = [C.Points(a) for a in arrays]
    outputs = [p.points.points for p in points]
    print(outputs)

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(C._current_precision)((1,-2))
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckSameTypesT(C._current_precision)((-2,-1))
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

def test_typeclass_same_v03():
    """Last input has another kind"""
    arrays = [np.arange(12, dtype='d').reshape(3,4) for i in range(5)]
    objects = [C.Points(a) for a in arrays[:-1]]
    objects.append(C.Histogram2d(np.arange(4), np.arange(5)))
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(C._current_precision)((1,-1), 'shape')
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckSameTypesT(C._current_precision)((1,-1), 'kind')
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

def test_typeclass_same_v04():
    """Last input has another edges"""
    objects=[C.Histogram2d(np.arange(4), np.arange(5)) for i in range(5)]
    objects.append(C.Histogram2d(np.arange(1,5), np.arange(5)))
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(C._current_precision)((1,-1), 'shape')
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckSameTypesT(C._current_precision)((1,-1),)
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

def test_typeclass_ndim_v01():
    objects=[C.Histogram(np.arange(6), np.arange(5)), C.Points(np.arange(5))]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckNdimT(C._current_precision)(1)
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckNdimT(C._current_precision)(2)
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

def test_typeclass_ndim_v02():
    objects=[C.Histogram2d(np.arange(6), np.arange(7)), C.Points(np.arange(12).reshape(3,4))]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckNdimT(C._current_precision)(2)
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckNdimT(C._current_precision)(1)
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

def test_typeclass_ndim_v03():
    objects=[C.Histogram(np.arange(6), np.arange(5)), C.Points(np.arange(5)), C.Histogram2d(np.arange(6), np.arange(7)), C.Points(np.arange(12).reshape(3,4))]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckNdimT(C._current_precision)(1, (0,1))
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckNdimT(C._current_precision)(2, (-2,-1))
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    res = obj.process_types();
    assert res

def test_typeclass_passtype():
    """Last input has another edges"""
    objects = [
            C.Histogram2d(np.arange(4), np.arange(5)),
            C.Histogram(np.arange(4)),
            C.Points(np.arange(12).reshape(3,4))
            ]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)
    for i in range(5):
        obj.add_output()

    dt1 = R.TypeClasses.PassTypeT(C._current_precision)((0,), (0,1))
    dt2 = R.TypeClasses.PassTypeT(C._current_precision)((1,), (2,-1))
    dt1.dump(); print()
    dt2.dump(); print()
    obj.add_typeclass(dt1)
    obj.add_typeclass(dt2)
    res = obj.process_types();
    assert res

    obj.print()
    dta = outputs[0].datatype()
    dtb = outputs[1].datatype()

    doutputs = obj.transformations.back().outputs
    assert doutputs[0].datatype()==dta
    assert doutputs[1].datatype()==dta
    assert doutputs[2].datatype()==dtb
    assert doutputs[3].datatype()==dtb
    assert doutputs[4].datatype()==dtb

def test_typeclass_passeach():
    """Last input has another edges"""
    objects = [
            C.Histogram2d(np.arange(4), np.arange(5)),
            C.Histogram(np.arange(4)),
            C.Points(np.arange(12).reshape(3,4))
            ]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)
    for i in range(5):
        obj.add_output()

    dt1 = R.TypeClasses.PassTypeT(C._current_precision)((2,), (0,1))
    dt2 = R.TypeClasses.PassEachTypeT(C._current_precision)((0,-1), (2,-1))
    dt1.dump(); print()
    dt2.dump(); print()
    obj.add_typeclass(dt1)
    obj.add_typeclass(dt2)
    res = obj.process_types();
    assert res

    obj.print()
    dta = outputs[0].datatype()
    dtb = outputs[1].datatype()
    dtc = outputs[2].datatype()

    doutputs = obj.transformations.back().outputs
    assert doutputs[0].datatype()==dtc
    assert doutputs[1].datatype()==dtc
    assert doutputs[2].datatype()==dta
    assert doutputs[3].datatype()==dtb
    assert doutputs[4].datatype()==dtc

if __name__ == "__main__":
    run_unittests(globals())

