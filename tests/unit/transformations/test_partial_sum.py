#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np 
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType

def arithmetic_progression(sequence):
    assert np.allclose(sequence[1:] - sequence[:-1], sequence[1]-sequence[0])
    return np.array([(i+1)*(upper + sequence[0])/2 for i, upper in enumerate(sequence)])
#
# Create the matrix
#
def test_partial_sum_hist():
    arr = np.arange(1, 13, dtype='d')
    edges = np.arange(0, 13, dtype='d')
    print( 'Input sequence' )
    print(arr)
    #
    # Create transformations
    #
    points = C.Histogram(edges, arr)
    partial_sum = R.PartialSum(0.)
    partial_sum.reduction << points

    res = partial_sum.reduction.data()
    expected = arithmetic_progression(arr)
    print("Expected")
    print(expected)
    print("Results from C++")
    print(res)
    assert np.allclose(expected, res)

def test_partial_sum_hist_with_start():
    arr = np.arange(1, 13, dtype='d')
    edges = np.arange(0, 13, dtype='d')
    validation = np.arange(3, 13, dtype='d')
    print( 'Input sequence' )
    print(arr)
    #
    # Create transformations
    #
    points = C.Histogram(edges, arr)
    partial_sum = R.PartialSum(2.)
    partial_sum.reduction << points

    res = partial_sum.reduction.data()
    expected = arithmetic_progression(validation)
    expected = np.concatenate((np.zeros(2), expected))
    print("Expected")
    print(expected)
    print("Results from C++")
    print(res)
    assert np.allclose(expected, res)

def test_partial_sum_points():
    arr = np.arange(0, 13, dtype='d')
    validation = arr
    #  validation = np.arange(2, 13, dtype='d')
    #  validation = np.concatenate((np.zeros(2), validation))
    print( 'Input sequence' )
    print(arr)
    #
    # Create transformations
    #
    points = C.Points(arr)
    partial_sum = R.PartialSum(0.)
    partial_sum.reduction << points

    res = partial_sum.reduction.data()
    expected = arithmetic_progression(validation)
    print("Expected")
    print(expected)
    print("Results from C++")
    print(res)
    assert np.allclose(expected, res)

def test_partial_sum_points_with_start():
    arr = np.arange(0, 13, dtype='d')
    validation = np.arange(2, 13, dtype='d')
    print( 'Input sequence' )
    print(arr)
    #
    # Create transformations
    #
    points = C.Points(arr)
    partial_sum = R.PartialSum(2.)
    partial_sum.reduction << points

    res = partial_sum.reduction.data()
    expected = arithmetic_progression(validation)
    expected = np.concatenate((np.zeros(2), expected))
    print("Expected")
    print(expected)
    print("Results from C++")
    print(res)
    assert np.allclose(expected, res)

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('\nRun ', fcn)
        glb[fcn]()

