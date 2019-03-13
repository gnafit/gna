#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import run_unittests, makefloat
from load import ROOT as R
import numpy as N

def arrayview_d_view(self):
    return N.frombuffer(self.data(), count=self.size(), dtype='d')

arrayview_d = R.arrayview('double')
arrayview_d.view = arrayview_d_view

def test_arrayview_constructor_view():
    a = N.arange(5, dtype='d')
    view = arrayview_d(a, a.size)

    assert view.size()==a.size
    assert not view.isOwner()

    vview = view.view()
    print('Python/C++', a, vview)

    assert (a==vview).all()
    assert view==view
    assert not view!=view

def test_arrayview_constructor():
    size=5
    a = N.arange(size, dtype='d')
    view = arrayview_d(size)

    assert view.size()==size
    assert view.isOwner()

    for i, v in enumerate(a):
        view[i]=v

    vview = view.view()
    print('C++', vview)

    assert (a==vview).all()

def test_arrayview_comparison():
    size=5
    a1 = N.arange(size, dtype='d')
    a2 = N.ascontiguousarray(a1[::-1])
    a3 = N.arange(size+1, dtype='d')
    view1a = arrayview_d(a1, a1.size)
    view1b = arrayview_d(a1, a1.size)
    view2  = arrayview_d(a2, a2.size)
    view3  = arrayview_d(a3, a3.size)

    assert     view1a==view1b
    assert not view1a!=view1b
    assert     view1a!=view2
    assert not view1a==view2
    assert     view1a!=view3
    assert not view1a==view3

    view4 = arrayview_d(view3.size())
    view4 = view3
    assert     view3==view4
    assert not view3!=view4

    view5 = arrayview_d(view3)
    assert     view3==view5
    assert not view3!=view5

def test_arrayview_vector():
    size=5
    from gna.constructors import stdvector
    a1 = N.arange(size, dtype='d')
    v1 = stdvector(a1)
    view1 = arrayview_d(a1, size)

    assert     view1==v1
    assert not view1!=v1

if __name__ == "__main__":
    run_unittests(globals())

