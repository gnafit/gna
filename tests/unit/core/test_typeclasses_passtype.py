#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
from gna import context
import numpy as np

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

    dt1 = R.TypeClasses.PassTypeT(context.current_precision())((0,), (0,1))
    dt2 = R.TypeClasses.PassTypeT(context.current_precision())((1,), (2,-1))
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
            C.Points(np.arange(20).reshape(4,5))
            ]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)
    for i in range(5):
        obj.add_output()

    dt1 = R.TypeClasses.PassTypeT(context.current_precision())((2,), (0,1))
    dt2 = R.TypeClasses.PassEachTypeT(context.current_precision())((0,-1), (2,-1))
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

def test_typeclass_passeach_02():
    """Pass with step 2"""
    objects = [
            C.Histogram2d(np.arange(4), np.arange(5)),
            C.Histogram(np.arange(4)),
            C.Points(np.arange(20).reshape(4,5))
            ]
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    map(obj.add_input, outputs)
    map(obj.add_input, outputs)
    for i in range(3):
        obj.add_output()

    dt1 = R.TypeClasses.PassEachTypeT(context.current_precision())((0,-1,2), (0,-1))
    dt1.dump(); print()
    obj.add_typeclass(dt1)
    res = obj.process_types();
    assert res

    obj.print()
    dta = outputs[0].datatype()
    dtb = outputs[1].datatype()
    dtc = outputs[2].datatype()

    doutputs = obj.transformations.back().outputs
    assert doutputs[0].datatype()==dta
    assert doutputs[1].datatype()==dtc
    assert doutputs[2].datatype()==dtb

if __name__ == "__main__":
    run_unittests(globals())

