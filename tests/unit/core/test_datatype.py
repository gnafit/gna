#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import *
from load import ROOT as R
import numpy as N
import gna.bindings.arrayview
from gna import context

def test_datatype_preallocated_v01():
    dt = R.DataType()
    buf = N.arange(10, dtype='d')

    dt.points().shape(buf.size).preallocated(buf)
    ptr = N.frombuffer(dt.buffer, count=1, dtype=dt.buffer.typecode).ctypes.data

    dt1 = R.DataType(dt)
    assert dt==dt1
    assert dt.buffer==dt1.buffer
    assert not dt.requiresReallocation(dt1)

    data=R.Data('double')(dt)
    assert data.type==dt
    assert data.type.buffer==dt.buffer
    assert not dt.requiresReallocation(data.type)

    ptr1 = N.frombuffer(data.buffer, count=1, dtype=data.buffer.typecode).ctypes.data
    assert ptr==ptr1

    buf1 = N.frombuffer(data.buffer, count=buf.size, dtype='d')
    assert (buf1==buf).all()

def test_datatype_preallocated_v02():
    dt = R.DataType()
    buf = N.arange(10, dtype='d')

    dt.points().shape(buf.size).preallocated(buf)

    dt1 = R.DataType()
    dt1.__assign__(dt)
    assert dt1==dt
    assert dt.requiresReallocation(dt1)

if __name__ == "__main__":
    run_unittests(globals())

