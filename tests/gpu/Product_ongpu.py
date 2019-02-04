#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np

m11 = np.ones((1,), dtype='d')
m34 = np.ones((3,4), dtype='d')
m33 = np.ones((3,3), dtype='d')

rup   = np.arange(12.0, dtype='d').reshape(3,4)
rdown = np.arange(12.0, dtype='d')[::-1].reshape(3,4)

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

    points = [C.Points(array) for array in arrays]
    prod = C.Product([p.points.points for p in points])

    prod.product.switchFunction("gpu")
    calc =  prod.single().data()
    print('Result', calc, end='\n\n')

    assert (calc==truth).all()

def test_01():
    check_product([m11])

def test_01a():
    check_product([m11*2, m11*3, m11*4])

def test_01b():
    check_product([m11*2, m11*0])

def test_02():
    check_product([m34])

def test_02b():
    check_product([m34, m34*0])

def test_06():
    check_product([3.0*m34, 4.0*m34, 2.0*m34])

def test_07():
    check_product([rup, rdown])

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('Run ', fcn)
        glb[fcn]()
