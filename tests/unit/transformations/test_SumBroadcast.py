#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *

sum_broadcast_mat_m11 = np.ones((1,), dtype='d')
sum_broadcast_mat_m34 = np.ones((3,4), dtype='d')
sum_broadcast_mat_m33 = np.ones((3,3), dtype='d')

sum_broadcast_rup   = np.arange(12.0, dtype='d').reshape(3,4)
sum_broadcast_rdown = np.arange(12.0, dtype='d')[::-1].reshape(3,4)

def check_sum_broadcast(arrays):
    print('Test ', len(arrays), ':', sep='')
    for array in arrays:
        print(array)
    print()

    truth=0.0
    for a in arrays:
        if a.size==1:
            truth+=a[0]
        else:
            truth+=a

    print('Truth:\n', truth, end='\n\n')

    points = [C.Points(array) for array in arrays]
    sum = C.SumBroadcast(outputs=[p.points.points for p in points])

    calc =  sum.single().data()
    print('Result', calc, end='\n\n')

    assert (calc==truth).all()

def test_sum_broadcast_01():
    check_sum_broadcast([sum_broadcast_mat_m11])

def test_sum_broadcast_01a():
    check_sum_broadcast([sum_broadcast_mat_m11*2, sum_broadcast_mat_m11*3, sum_broadcast_mat_m11*4])

def test_sum_broadcast_01b():
    check_sum_broadcast([sum_broadcast_mat_m11*2, sum_broadcast_mat_m11*0])

def test_sum_broadcast_02():
    check_sum_broadcast([sum_broadcast_mat_m34])

def test_sum_broadcast_02b():
    check_sum_broadcast([sum_broadcast_mat_m34, sum_broadcast_mat_m34*0])

def test_sum_broadcast_03():
    check_sum_broadcast([2.0*sum_broadcast_mat_m11, 3.0*sum_broadcast_mat_m34])

def test_sum_broadcast_04():
    check_sum_broadcast([3.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m11])

def test_sum_broadcast_03():
    check_sum_broadcast([2.0*sum_broadcast_mat_m11, 3.0*sum_broadcast_mat_m34, 4.0*sum_broadcast_mat_m34])

def test_sum_broadcast_04():
    check_sum_broadcast([3.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m11, 4.0*sum_broadcast_mat_m34])

def test_sum_broadcast_05():
    check_sum_broadcast([3.0*sum_broadcast_mat_m34, 4.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m11])

def test_sum_broadcast_06():
    check_sum_broadcast([3.0*sum_broadcast_mat_m34, 4.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m34])

def test_sum_broadcast_07():
    check_sum_broadcast([sum_broadcast_rup, sum_broadcast_rdown])

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('Run ', fcn)
        glb[fcn]()
