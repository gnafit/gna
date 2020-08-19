#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import *
from load import ROOT as R
import numpy as N
import gna.bindings.arrayview
from gna.bindings import __root_version__
from gna import context

@floatcopy(globals())
def test_arrayview_constructor_view():
    a = N.arange(5, dtype=context.current_precision_short())
    view = R.arrayview(context.current_precision())(a, a.size)

    assert view.size()==a.size
    assert not view.isOwner()

    vview = view.view()
    print('Python/C++', a, vview)

    assert (a==vview).all()
    assert view==view
    assert not view!=view

@floatcopy(globals())
def test_arrayview_constructor():
    size=5
    import ctypes
    san_size = ctypes.c_int(5)
    a = N.arange(size, dtype=context.current_precision_short())
    print("in failing test! Precision is {}".format(context.current_precision()))
    view = R.arrayview[context.current_precision()](size)

    assert view.size()==size
    assert view.isOwner()

    for i, v in enumerate(a):
        view[i]=v

    vview = view.view()
    print('C++', vview)

    assert (a==vview).all()

@floatcopy(globals())
def test_arrayview_comparison():
    size=5
    a1 = N.arange(size, dtype=context.current_precision_short())
    a2 = N.ascontiguousarray(a1[::-1])
    a3 = N.arange(size+1, dtype=context.current_precision_short())
    view1a = R.arrayview(context.current_precision())(a1, a1.size)
    view1b = R.arrayview(context.current_precision())(a1, a1.size)
    view2  = R.arrayview(context.current_precision())(a2, a2.size)
    view3  = R.arrayview(context.current_precision())(a3, a3.size)

    assert     view1a==view1b
    assert not view1a!=view1b
    assert     view1a!=view2
    assert not view1a==view2
    assert     view1a!=view3
    assert not view1a==view3

    view4 = R.arrayview(context.current_precision())(view3.size())
    view4 = view3
    assert     view3==view4
    assert not view3!=view4

    view5 = R.arrayview(context.current_precision())(view3)
    assert     view3==view5
    assert not view3!=view5

@floatcopy(globals())
def test_arrayview_vector():
    size=5
    from gna.constructors import stdvector
    a1 = N.arange(size, dtype=context.current_precision_short())
    v1 = stdvector(a1)
    view1 = R.arrayview(context.current_precision())(a1, size)

    assert     view1==v1
    assert not view1!=v1

@floatcopy(globals())
def test_arrayview_complex():
    a1 = R.arrayview(context.current_precision())(2)
    a1[0]=2
    a1[1]=3
    c1 = a1.complex()
    if __root_version__ >= '6.22/00':
        assert c1.real == 2.0
        assert c1.imag == 3.0
    else:
        assert c1.real() == 2.0
        assert c1.imag() == 3.0

if __name__ == "__main__":
    run_unittests(globals())
