#!/usr/bin/env python


from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
from gna import context
import numpy as np

def test_typeclass_kind_v01():
    objects=[C.Histogram(np.arange(6), np.arange(5)), C.Histogram2d(np.arange(6), np.arange(7))]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    list(map(obj.add_input, outputs))

    dt_points = R.TypeClasses.CheckKindT(context.current_precision())(1)
    R.SetOwnership(dt_points, False)
    dt_points.dump(); print()

    dt_hist = R.TypeClasses.CheckKindT(context.current_precision())(2)
    R.SetOwnership(dt_hist, False)
    dt_hist.dump(); print()

    obj.add_typeclass(dt_hist)
    res = obj.process_types();
    assert res

    obj.add_typeclass(dt_points)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

def test_typeclass_kind_v02():
    objects=[C.Points(np.arange(6)), C.Points(np.arange(6).reshape(3,2))]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    list(map(obj.add_input, outputs))

    dt_points = R.TypeClasses.CheckKindT(context.current_precision())(1)
    R.SetOwnership(dt_points, False)
    dt_points.dump(); print()

    dt_hist = R.TypeClasses.CheckKindT(context.current_precision())(2)
    R.SetOwnership(dt_hist, False)
    dt_hist.dump(); print()

    obj.add_typeclass(dt_points)
    res = obj.process_types();
    assert res

    obj.add_typeclass(dt_hist)
    print('Exception expected: ',end='')
    res = obj.process_types();
    assert not res

if __name__ == "__main__":
    run_unittests(globals())
