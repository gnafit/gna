#!/usr/bin/env python


from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
from gna import context
import numpy as np
import pytest

@pytest.mark.parametrize('order', (0, 1))
def test_typeclass_passtype_priority(order):
    """Last input has another edges"""
    data = np.arange(20, dtype='d')
    if order==0:
        objects = [
                C.Points(data[:1]),       # 0
                C.Histogram(data[:2]),    # 1
                C.Points(data[:5]),       # 2
                C.Histogram(data[:6]),    # 3
                C.Points(data[:10]),      # 4
                C.Histogram(data[:11]),   # 5
                ]
    elif order==1:
        objects = [
                C.Points(data[:1]),       # 0
                C.Points(data[:5]),       # 1
                C.Histogram(data[:2]),    # 2
                C.Histogram(data[:6]),    # 3
                C.Points(data[:10]),      # 4
                C.Histogram(data[:11]),   # 5
                ]
    else:
        assert False
    outputs = [p.single() for p in objects]

    obj = C.DummyType()
    for i in range(4):
        obj.add_output(f'{i}')
    for i, out in enumerate(outputs):
        obj.add_input(out, f'input_{i}')

    dt1 = R.TypeClasses.PassTypePriorityT(context.current_precision())((0,-1), (0,0))
    dt2 = R.TypeClasses.PassTypePriorityT(context.current_precision())((0,-1), (1,1), True, False)
    dt3 = R.TypeClasses.PassTypePriorityT(context.current_precision())((0,-1), (2,2), False, True)
    dt4 = R.TypeClasses.PassTypePriorityT(context.current_precision())((0,-1), (3,3), False, False)
    dts=[dt1, dt2, dt3, dt4]
    for dt in dts:
        R.SetOwnership(dt, False)
        dt.dump(); print()
        obj.add_typeclass(dt)
    res = obj.process_types();
    assert res

    obj.print()
    dt = outputs[0].datatype()

    dtypes = [out.datatype() for out in outputs]
    doutputs = obj.transformations.back().outputs
    if order==0:
        assert doutputs[0].datatype()==dtypes[3]
        assert doutputs[1].datatype()==dtypes[1]
        assert doutputs[2].datatype()==dtypes[2]
        assert doutputs[3].datatype()==dtypes[0]
    elif order==1:
        assert doutputs[0].datatype()==dtypes[3]
        assert doutputs[1].datatype()==dtypes[2]
        assert doutputs[2].datatype()==dtypes[1]
        assert doutputs[3].datatype()==dtypes[0]

