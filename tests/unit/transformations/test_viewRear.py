#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import env
from gna import context

@floatcopy(globals(), addname=True)
def test_viewrear_points_01_start_len(function_name):
    """Test ViewRear on Points (start, len)"""
    arr = N.zeros(12, dtype=context.current_precision_short())
    ns = env.env.globalns(function_name)
    names=[]
    for i in range(arr.size):
        name='val_%02i'%i
        names.append(name)
        ns.defparameter( name, central=i, fixed=True, label='Value %i'%i )

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        start, len = rng

        pview = arr[start:start+len]
        points = C.Points(arr)
        view = C.ViewRear(points, start, len);

        cnames = names[start:start+len]
        with ns:
            vararray = C.VarArray(cnames)

        vararray >> view.view.rear

        print('Range', rng)
        for ichange, iname in enumerate(['']+cnames, -1):
            if iname:
                print('  Change', ichange)
                par=ns[iname]
                par.set(par.value()+1.0)

            res0 = vararray.single().data()
            res = view.view.result.data()
            expect = arr.copy()
            for i in range(start, start+len):
                expect[i] = ns[names[i]].value()

            expect0 = []
            for i in range(start, start+len):
                expect0.append(ns[names[i]].value())

            print('    Result 0', res0)
            print('    Expect 0', expect0)
            print('    Result', res)
            print('    Expect', expect)
            print('    Original data', points.single().data())
            print('    Original data (expect)', arr)

            assert (res==expect).all()
            assert (res0==expect0).all()
            assert (res[start:start+len]==res0).all()
            assert (points.single().data()==arr).all()
            print()

@floatcopy(globals(), addname=True)
def test_viewrear_points_02_start(function_name):
    """Test ViewRear on Points (start) [no len]"""
    arr = N.zeros(12, dtype=context.current_precision_short())
    ns = env.env.globalns(function_name)
    names=[]
    for i in range(arr.size):
        name='val_%02i'%i
        names.append(name)
        ns.defparameter( name, central=i, fixed=True, label='Value %i'%i )

    for start in range(arr.size):
        pview = arr[start:]
        points = C.Points(arr)
        view = C.ViewRear(points, start);

        cnames = names[start:]
        with ns:
            vararray = C.VarArray(cnames)

        vararray >> view.view.rear

        print('Start', start)
        for ichange, iname in enumerate(['']+cnames, -1):
            if iname:
                print('  Change', ichange, iname)
                par=ns[iname]
                par.set(par.value()+1.0)

            res0 = vararray.single().data()
            res = view.view.result.data()
            expect = arr.copy()
            for i, name in enumerate(cnames, start):
                expect[i] = ns[name].value()

            expect0 = []
            for i, name in enumerate(cnames, start):
                expect0.append(ns[name].value())

            print('    Result 0', res0)
            print('    Expect 0', expect0)
            print('    Result', res)
            print('    Expect', expect)
            print('    Original data', points.single().data())
            print('    Original data (expect)', arr)

            assert (res==expect).all()
            assert (res0==expect0).all()
            assert (res[start:]==res0).all()
            assert (points.single().data()==arr).all()
            print()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_viewrear_points_03_auto_fromhist():
    """Test ViewRear on Histogram (determine start and length)"""
    edges = N.arange(13, dtype='d')
    arr = N.zeros(edges.size-1, dtype=context.current_precision_short())
    points_main = C.Points(arr)

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        print('Range', rng)
        hist_main = C.Histogram(edges, arr)
        start, len = rng
        pedges = edges[start:start+len+1]
        arr_sub = N.arange(start, start+len, dtype=arr.dtype)
        hist_sub=C.Histogram(pedges, arr_sub, True)

        points_sub=C.Points(arr_sub, True)

        view = C.ViewRear();
        view.determineOffset(hist_main, hist_sub)
        points_main >> view.view.original
        points_sub >> view.view.rear

        res = view.view.result.data()
        expect = arr.copy()
        expect[start:start+len]=arr_sub
        print('Result', res)
        print('Expect', expect)
        print()
        assert (res==expect).all()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_viewrear_points_04_auto_fromhist_edges():
    """Test ViewRear on Histogram (determine start and length)"""
    edges = N.arange(13, dtype='d')
    arr = N.zeros(edges.size, dtype=context.current_precision_short())
    points_main = C.Points(arr)

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        print('Range', rng)
        hist_main = C.Histogram(edges, arr[:-1])
        start, len = rng
        pedges = edges[start:start+len+1]
        arr_sub = N.arange(start, start+len+1, dtype=arr.dtype)
        hist_sub=C.Histogram(pedges, arr_sub[:-1], True)

        points_sub=C.Points(arr_sub, True)

        view = C.ViewRear()
        view.determineOffset(hist_main, hist_sub, True)
        points_main >> view.view.original
        points_sub >> view.view.rear

        res = view.view.result.data()
        expect = arr.copy()
        expect[start:start+len+1]=arr_sub
        print('Result', res)
        print('Expect', expect)
        print()
        assert (res==expect).all()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_viewrear_hist_01_start_len():
    """Test ViewRear on Histogram (start, len)"""
    edges = N.arange(13, dtype='d')
    arr = N.zeros(edges.size-1, dtype=context.current_precision_short())

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        hist_main = C.Histogram(edges, arr)
        start, len = rng

        view = C.ViewRear(hist_main, start, len);

        pedges = edges[start:start+len+1]
        arr_sub = N.arange(start, start+len, dtype=arr.dtype)
        hist_sub=C.Histogram(pedges, arr_sub, True)
        hist_sub >> view.view.rear

        res = view.view.result.data()
        expect_edges = N.array(view.view.result.datatype().edges)
        expect = arr.copy()
        expect[start:start+len]=arr_sub
        print('Range', rng)
        print('Result', res)
        print('Expect', expect)
        print('Result (edges)', edges)
        print('Expect (edges)', expect_edges)
        print()
        assert (res==expect).all()
        assert (edges==expect_edges).all()
        assert (hist_main.single().data()==arr).all()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_viewrear_hist_02_auto():
    """Test ViewRear on Histogram (determine start and length)"""
    edges = N.arange(13, dtype='d')
    arr = N.zeros(edges.size-1, dtype=context.current_precision_short())

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        print('Range', rng)
        hist_main = C.Histogram(edges, arr)
        start, len = rng
        pedges = edges[start:start+len+1]
        arr_sub = N.arange(start, start+len, dtype=arr.dtype)
        hist_sub=C.Histogram(pedges, arr_sub, True)

        view = C.ViewRear(hist_main);
        hist_sub >> view.view.rear

        res = view.view.result.data()
        expect_edges = N.array(view.view.result.datatype().edges)
        expect = arr.copy()
        expect[start:start+len]=arr_sub
        print('Result', res)
        print('Expect', expect)
        print('Result (edges)', edges)
        print('Expect (edges)', expect_edges)
        print()
        assert (res==expect).all()
        assert (edges==expect_edges).all()
        assert (hist_main.single().data()==arr).all()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_viewrear_hist_03_start_len_threshold():
    """Test ViewRear on Histogram (start, len), enable threshold"""
    edges = N.arange(13, dtype='d')
    arr = N.zeros(edges.size-1, dtype=context.current_precision_short())

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        hist_main = C.Histogram(edges, arr)
        start, len = rng

        view = C.ViewRear(hist_main, start, len);
        view.allowThreshold()

        pedges = edges[start:start+len+1].copy()
        print('Original edges', pedges)
        pedges[0]  = 0.5*(pedges[:2].sum())
        pedges[-1] = 0.5*(pedges[-2:].sum())
        print('Modified edges', pedges)
        arr_sub = N.arange(start, start+len, dtype=arr.dtype)
        hist_sub=C.Histogram(pedges, arr_sub, True)
        hist_sub >> view.view.rear

        res = view.view.result.data()
        expect_edges = N.array(view.view.result.datatype().edges)
        expect = arr.copy()
        expect[start:start+len]=arr_sub
        print('Range', rng)
        print('Result', res)
        print('Expect', expect)
        print('Result (edges)', edges)
        print('Expect (edges)', expect_edges)
        print()
        assert (res==expect).all()
        assert (edges==expect_edges).all()
        assert (hist_main.single().data()==arr).all()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_viewrear_hist_04_auto_threshold():
    """Test ViewRear on Histogram (auto start and len), enable threshold"""
    edges = N.arange(13, dtype='d')
    arr = N.zeros(edges.size-1, dtype=context.current_precision_short())

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6)]
    for rng in ranges:
        hist_main = C.Histogram(edges, arr)
        start, len = rng

        pedges = edges[start:start+len+1].copy()
        print('Original edges', pedges)
        pedges[0]  = 0.5*(pedges[:2].sum())
        pedges[-1] = 0.5*(pedges[-2:].sum())
        print('Modified edges', pedges)
        arr_sub = N.arange(start, start+len, dtype=arr.dtype)
        hist_sub=C.Histogram(pedges, arr_sub, True)

        view = C.ViewRear(hist_main);
        view.allowThreshold()
        hist_sub >> view.view.rear

        res = view.view.result.data()
        expect_edges = N.array(view.view.result.datatype().edges)
        expect = arr.copy()
        expect[start:start+len]=arr_sub
        print('Range', rng)
        print('Result', res)
        print('Expect', expect)
        print('Result (edges)', edges)
        print('Expect (edges)', expect_edges)
        print()
        assert (res==expect).all()
        assert (edges==expect_edges).all()
        assert (hist_main.single().data()==arr).all()

@floatcopy(globals(), addname=True)
def test_viewrear_points_vars_01(function_name):
    """Test ViewRear on Points (start, len), check taintflag propagation"""
    arr = N.arange(12, dtype=context.current_precision_short())
    ns = env.env.globalns(function_name)
    names=[]
    for i in range(arr.size):
        name='val_%02i'%i
        names.append(name)
        ns.defparameter( name, central=i, fixed=True, label='Value %i'%i )

    fill_value = -1.0

    ranges = [ (0, 3), (0, 12), (1, 3), (6, 6), (6, 1)]
    for rng in ranges:
        start, len = rng

        pview = arr[start:start+len]
        points = C.Points(arr)
        view = C.ViewRear(points, start, len, fill_value);

        cnames = names[start:start+len]
        with ns:
            vararray = C.VarArray(cnames)

        vararray >> view.view.rear

        print('Range', rng)
        for ichange, iname in enumerate(['']+cnames, -1):
            if iname:
                print('  Change', ichange)
                par=ns[iname]
                par.set(par.value()+1.0)

            res0 = vararray.single().data()
            res = view.view.result.data()
            expect = N.zeros_like(arr)+fill_value
            for i in range(start, start+len):
                expect[i] = ns[names[i]].value()

            expect0 = []
            for i in range(start, start+len):
                expect0.append(ns[names[i]].value())

            print('    Result 0', res0)
            print('    Expect 0', expect0)
            print('    Result', res)
            print('    Expect', expect)
            assert (res==expect).all()
            assert (res0==expect0).all()
            assert (res[start:start+len]==res0).all()
            assert (points.single().data()==arr).all()
            print()

if __name__ == "__main__":
    run_unittests(globals())

