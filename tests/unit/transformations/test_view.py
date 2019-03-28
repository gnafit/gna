#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class

and test the matrix memory ordering"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *

@floatcopy(globals())
def test_view_01():
    arr = N.arange(12, dtype='d')
    points = C.Points(arr)

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        start, len = rng
        pview = arr[start:start+len]
        view = C.View(points, start, len);

        res = view.view.view.data()
        print('Range', rng)
        print('Result', res)
        print('Expect', pview)
        print()
        assert (res==pview).all()

# @floatcopy(globals())
def test_view_02():
    arr = N.arange(13, dtype='d')
    points = C.Histogram(arr, arr[:-1])

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        start, len = rng
        pview  = arr[start:start+len]
        pedges = arr[start:start+len+1]
        view = C.View(points, start, len);

        res = view.view.view.data()
        edges = N.array(view.view.view.datatype().edges)
        print('Range', rng)
        print('Result', res)
        print('Expect', pview)
        print('Result (edges)', edges)
        print('Expect (edges)', pedges)
        print()
        assert (res==pview).all()
        assert (edges==pedges).all()

if __name__ == "__main__":
    run_unittests(globals())

