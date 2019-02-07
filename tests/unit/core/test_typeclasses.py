#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import run_unittests
from gna import constructors as C
import numpy as np

def check(r, size, expect):
    what = list(r.vector(size, False))
    print('Check:', what, expect)
    assert what==expect

def test_range_01():
    r = R.TypeClasses.Range(0)
    r.dump()

    check(r, 1, [0])
    check(r, 2, [0])
    check(r, 3, [0])

def test_range_02():
    r = R.TypeClasses.Range(1)
    r.dump()

    check(r, 1, [])
    check(r, 2, [1])
    check(r, 3, [1])

def test_range_03():
    r = R.TypeClasses.Range(0,2)
    r.dump()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [0,1,2])

def test_range_04():
    r = R.TypeClasses.Range(0,-1)
    r.dump()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [0,1,2,3])

def test_range_05():
    r = R.TypeClasses.Range(2,-1)
    r.dump()

    check(r, 1, [])
    check(r, 2, [])
    check(r, 3, [2])
    check(r, 4, [2,3])
    check(r, 5, [2,3,4])
    check(r, 6, [2,3,4,5])

def test_range_06():
    r = R.TypeClasses.Range(-3,-1)
    r.dump()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [1,2,3])
    check(r, 5, [2,3,4])

def test_range_07():
    r = R.TypeClasses.Range(-3,2)
    r.dump()

    check(r, 1, [0])
    check(r, 2, [0,1])
    check(r, 3, [0,1,2])
    check(r, 4, [1,2])
    check(r, 5, [2])
    check(r, 6, [])

def test_typeclass_same_v01():
    arrays = [np.arange(12, dtype='d').reshape(3,4) for i in range(5)]
    points = [C.Points(a) for a in arrays]
    outputs = [p.points.points for p in points]

    precision = C._current_precision
    print("Make dummy")
    obj = C.DummyType()
    map(obj.add_input, outputs)

    dt = R.TypeClasses.CheckSameTypesT(precision)
    # obj.add_dataclass(dt)

    # import IPython; IPython.embed()

if __name__ == "__main__":
    run_unittests(globals())
