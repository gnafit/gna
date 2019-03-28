#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
import numpy as np

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

if __name__ == "__main__":
    run_unittests(globals())

