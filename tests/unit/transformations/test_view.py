#!/usr/bin/env python

"""Check Points class

and test the matrix memory ordering"""

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import env

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

# @floatcopy(globals()) # uncomment after porting the histogram
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

@floatcopy(globals(), addname=True)
def test_view_03(function_name):
    ns = env.env.globalns(function_name)
    names=[]
    for i in range(12):
        name='val_%02i'%i
        names.append(name)
        ns.defparameter( name, central=i, fixed=True, label='Value %i'%i )

    with ns:
        vararray = C.VarArray(names)

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        print('Range', rng)
        start, len = rng
        cnames = names[start:start+len]

        view = C.View(vararray, start, len);

        for ichange, iname in enumerate(['']+cnames, -1):
            if iname:
                print('  Change', ichange)
                par=ns[iname]
                par.set(par.value()+1.0)

            expect = vararray.single().data()
            res = view.view.view.data()

            print('    Result', res)
            print('    Expect 0', expect)
            expect = expect[start:start+len]
            print('    Expect', expect)
            assert (res==expect).all()
            print()

if __name__ == "__main__":
    run_unittests(globals())

