#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *

sum_mat_m11 = np.ones((1,), dtype='d')
sum_mat_m34 = np.ones((3,4), dtype='d')
sum_mat_m33 = np.ones((3,3), dtype='d')

sum_rup   = np.arange(12.0, dtype='d').reshape(3,4)
sum_rdown = np.arange(12.0, dtype='d')[::-1].reshape(3,4)

def check_sum(arrays):
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
    sum = C.Sum(outputs=[p.points.points for p in points])

    calc =  sum.single().data()
    print('Result', calc, calc.dtype, end='\n\n')

    assert (calc==truth).all()

@clones(globals(), float=True, gpu=True)
def test_sum_01():
    check_sum([sum_mat_m11])

@clones(globals(), float=True, gpu=True)
def test_sum_01a():
    check_sum([sum_mat_m11*1, sum_mat_m11*3, sum_mat_m11*4])

@clones(globals(), float=True, gpu=True)
def test_sum_01b():
    check_sum([sum_mat_m11*2, sum_mat_m11*0])

@clones(globals(), float=True, gpu=True)
def test_sum_02():
    check_sum([sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_02b():
    check_sum([sum_mat_m34, sum_mat_m34*0])

@clones(globals(), float=True, gpu=True)
def test_sum_03():
    check_sum([3.0*sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_04():
    check_sum([3.0*sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_03():
    check_sum([3.0*sum_mat_m34, 4.0*sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_04():
    check_sum([3.0*sum_mat_m34, 4.0*sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_05():
    check_sum([3.0*sum_mat_m34, 4.0*sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_06():
    check_sum([3.0*sum_mat_m34, 4.0*sum_mat_m34, 2.0*sum_mat_m34])

@clones(globals(), float=True, gpu=True)
def test_sum_07():
    check_sum([sum_rup, sum_rdown])

if __name__ == "__main__":
    run_unittests(globals())
