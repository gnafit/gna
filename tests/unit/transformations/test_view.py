#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import env

def test_view_01():
    arr = N.arange(12, dtype='d')
    points = C.Points(arr)

    ranges = [ None, (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        if rng:
            start, len = rng
            pview = arr[start:start+len]
            view = C.View(points, start, len);
        else:
            pview = arr
            view = C.View(points);

        res = view.view.view.data()
        print('Range', rng)
        print('Result', res)
        print('Expect', pview)
        print()
        assert (res==pview).all()

def test_view_01a():
    arr = N.arange(12, dtype='d').reshape(3,4)
    points = C.Points(arr)

    ranges = [None]
    for rng in ranges:
        if rng:
            start, len = rng
            pview = arr[start:start+len]
            view = C.View(points, start, len);
        else:
            pview = arr
            view = C.View(points);

        res = view.view.view.data()
        print('Range', rng)
        print('Result', res)
        print('Expect', pview)
        print()
        assert (res==pview).all()

def test_view_01b_chain():
    arr = N.arange(12, dtype='d')
    points = C.Points(arr)

    ranges = [ None, (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        if rng:
            start, len = rng
            pview = arr[start:start+len]
            view1 = C.View(points, start, len);
        else:
            pview = arr
            view1 = C.View(points);

        view2 = C.View(view1.single())

        res = view2.view.view.data()
        print('Range', rng)
        print('Result', res)
        print('Expect', pview)
        print()
        assert (res==pview).all()


def test_view_02():
    arr = N.arange(13, dtype='d')
    points = C.Histogram(arr, arr[:-1])

    ranges = [ None, (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        if rng:
            start, len = rng
            pview  = arr[start:start+len]
            pedges = arr[start:start+len+1]
            view = C.View(points, start, len);
        else:
            pview  = arr[:-1]
            pedges = arr
            view = C.View(points);

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

@passname
def test_view_03(function_name):
    ns = env.env.globalns(function_name)
    names=[]
    for i in range(12):
        name='val_%02i'%i
        names.append(name)
        ns.defparameter( name, central=i, fixed=True, label='Value %i'%i )

    with ns:
        vararray = C.VarArray(names)

    expect = vararray.single().data()

    ranges = [ None, (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        print('Range', rng)
        if rng:
            start, len = rng
            cnames = names[start:start+len]
            view = C.View(vararray, start, len);
            cexpect = expect[start:start+len]
        else:
            cnames = names
            view = C.View(vararray);
            cexpect = expect

        view2 = C.View(view.single())

        for ichange, iname in enumerate(['']+cnames, -1):
            if iname:
                print('  Change', ichange)
                par=ns[iname]
                par.set(par.value()+1.0)

            res2 = view2.view.view.data()
            res = view.view.view.data()

            print('    Result', res)
            print('    Result 2', res2)
            print('    Expect', cexpect)
            assert (res==cexpect).all()
            assert (res2==cexpect).all()
            print()

if __name__ == "__main__":
    run_unittests(globals())

