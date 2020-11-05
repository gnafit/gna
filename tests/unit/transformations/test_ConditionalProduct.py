#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *
from gna.env import env

condproduct_mat_m11 = np.ones((1,), dtype='d')
condproduct_mat_m34 = np.ones((3,4), dtype='d')
condproduct_mat_m33 = np.ones((3,3), dtype='d')

condproduct_rup   = np.arange(12.0, dtype='d').reshape(3,4)
condproduct_rdown = np.arange(12.0, dtype='d')[::-1].reshape(3,4)

def check_condproduct(function_name, arrays):
    print('Test ', function_name, len(arrays), ':', sep='')
    for array in arrays:
        print(array)
    print()

    nprod = len(arrays)-1
    truth1=1.0
    truth2=1.0
    for i, a in enumerate(arrays):
        truth1*=a

        if i<nprod:
            truth2*=a

    ns = env.globalns(function_name)
    condition = ns.defparameter('condition', central=1.0, fixed=True)
    points = [C.Points(array) for array in arrays]
    with ns:
        prod = C.ConditionalProduct(nprod, 'condition', outputs=[p.points.points for p in points])

    calc1 =  prod.single().data().copy()
    print('Result (1)', condition.value(), calc1, end='\n\n')
    condition.set(0.0)
    calc2 =  prod.single().data().copy()
    print('Result (0)', condition.value(), calc2, end='\n\n')

    assert (calc1==truth1).all()
    assert (calc2==truth2).all()

@passname
def test_condproduct_01a(function_name):
    check_condproduct(function_name, [condproduct_mat_m11*2, condproduct_mat_m11*3, condproduct_mat_m11*4])

@passname
def test_condproduct_01b(function_name):
    check_condproduct(function_name, [condproduct_mat_m11*2, condproduct_mat_m11*0])

@passname
def test_condproduct_02b(function_name):
    check_condproduct(function_name, [condproduct_mat_m34, condproduct_mat_m34*0])

@passname
def test_condproduct_03(function_name):
    check_condproduct(function_name, [3.0*condproduct_mat_m34, 4.0*condproduct_mat_m34])

@passname
def test_condproduct_04(function_name):
    check_condproduct(function_name, [3.0*condproduct_mat_m34, 4.0*condproduct_mat_m34])

@passname
def test_condproduct_05(function_name):
    check_condproduct(function_name, [3.0*condproduct_mat_m34, 4.0*condproduct_mat_m34])

@passname
def test_condproduct_06(function_name):
    check_condproduct(function_name, [3.0*condproduct_mat_m34, 4.0*condproduct_mat_m34, 2.0*condproduct_mat_m34])

@passname
def test_condproduct_07(function_name):
    check_condproduct(function_name, [condproduct_rup, condproduct_rdown])

if __name__ == "__main__":
    run_unittests(globals())
