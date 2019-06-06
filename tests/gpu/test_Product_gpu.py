#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *
from gna import context, bindings

product_mat_m11 = np.ones((1,), dtype='d')
product_mat_m34 = np.ones((3,4), dtype='d')
product_mat_m33 = np.ones((3,3), dtype='d')

product_rup   = np.arange(12.0, dtype='d').reshape(3,4)
product_rdown = np.arange(12.0, dtype='d')[::-1].reshape(3,4)

def check_product(arrays):
    print('Test ', len(arrays), ':', sep='')
    for array in arrays:
        print(array)
    print()

    truth=1.0
    for a in arrays:
        if a.size==1:
            truth*=a[0]
        else:
            truth*=a

    print('Truth:\n', truth, end='\n\n')
    ndata =100
    with context.manager(ndata) as manager:
        points = [C.Points(array) for array in arrays]
        prod = C.Product([p.points.points for p in points])

        calc =  prod.single().data()
        print('Result', calc, end='\n\n')

    assert (calc==truth).all()

def test_product_01():
    check_product([product_mat_m11])

def test_product_01a():
    check_product([product_mat_m11*2, product_mat_m11*3, product_mat_m11*4])

def test_product_01b():
    check_product([product_mat_m11*2, product_mat_m11*0])

def test_product_02():
    check_product([product_mat_m34])

def test_product_02b():
    check_product([product_mat_m34, product_mat_m34*0])

def test_product_03():
    check_product([2.0*product_mat_m11, 3.0*product_mat_m34])


def test_product_03():
    check_product([3.0*product_mat_m34, 4.0*product_mat_m34])

def test_product_06():
    check_product([3.0*product_mat_m34, 4.0*product_mat_m34, 2.0*product_mat_m34])

def test_product_07():
    check_product([product_rup, product_rdown])

if __name__ == "__main__":
    run_unittests(globals())
