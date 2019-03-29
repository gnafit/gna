#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
from gna import context
import numpy as np

def test_typeclass_setpoints_v01():
    sizes = (1, 2, 3)

    for size in sizes:
        obj = C.DummyType()
        obj.add_output('out')

        dt = R.TypeClasses.SetPointsT(context.current_precision())(size)
        dt.dump(); print()
        obj.add_typeclass(dt)
        res = obj.process_types();
        assert res

        dtype = obj.dummytype.out.datatype()
        assert dtype.kind==1
        assert dtype.shape.size()==1
        assert dtype.shape[0]==size

def test_typeclass_setpoints_v01():
    sizes = ((1,2), (2,3), (5,3))

    for size in sizes:
        obj = C.DummyType()
        obj.add_output('out')

        dt = R.TypeClasses.SetPointsT(context.current_precision())(*size)
        dt.dump(); print()
        obj.add_typeclass(dt)
        res = obj.process_types();
        assert res

        dtype = obj.dummytype.out.datatype()
        assert dtype.kind==1
        assert dtype.shape.size()==2
        assert dtype.shape[0]==size[0]
        assert dtype.shape[1]==size[1]

if __name__ == "__main__":
    run_unittests(globals())

