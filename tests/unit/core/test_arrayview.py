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
allocator_d = R.arrayviewAllocatorSimple('double')
allocator_d.view = arrayview_d_view

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

def test_arrayview_complex():
    a1 = arrayview_d(2)
    a1[0]=2
    a1[1]=3
    c1 = a1.complex()
    assert c1.real()==2.0
    assert c1.imag()==3.0

# def test_arrayview_allocation():
    # nitems, ndata = 5, 15
    # allocator = allocator_d(ndata)

    # arrays = []
    # for i in range(1, nitems):
        # array = arrayview_d(i, allocator)
        # for j in range(i):
            # array[j]=j
        # print(i, array.view())
        # arrays.append(array)

    # data = allocator.view()
    # print('Arrays:', [array.view() for array in arrays])
    # print('Data:', data)

if __name__ == "__main__":
    run_unittests(globals())

