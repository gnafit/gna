#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
from gna import context
import numpy as np

@floatcopy(globals())
def test_typeclass_same_v01():
    """All inputs have same types"""
    arrays = [np.arange(12, dtype='d').reshape(3,4) for i in range(5)]
    points = [C.Points(a) for a in arrays]
    outputs = [p.points.points for p in points]

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(context.current_precision())((0,-1))
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

@floatcopy(globals())
def test_typeclass_same_v02():
    """Last input has another shape"""
    arrays = [np.arange(12, dtype='d').reshape(3,4) for i in range(5)]
    arrays[-1]=arrays[-1].reshape(4,3)
    points = [C.Points(a) for a in arrays]
    outputs = [p.points.points for p in points]
    print(outputs)

    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(context.current_precision())((1,-2))
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckSameTypesT(context.current_precision())((-2,-1))
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

    dt = R.TypeClasses.CheckSameTypesT(context.current_precision())((1,-1), 'shape')
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckSameTypesT(context.current_precision())((1,-1), 'kind')
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

    dt = R.TypeClasses.CheckSameTypesT(context.current_precision())((1,-1), 'shape')
    dt.dump(); print()
    obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    dt1 = R.TypeClasses.CheckSameTypesT(context.current_precision())((1,-1),)
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

if __name__ == "__main__":
    run_unittests(globals())

