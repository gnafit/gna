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

#
# Create the matrix
#
def test_partial_sum():
    arr = np.arange(12, dtype='d')

    print( 'Input matrix (numpy)' )
    print(arr)

    def arithmetic_progression(start, last, n):
        return start
    #
    # Create transformations
    #
    points = C.Points(arr)
    partial_sum = R.PartialSum()
    partial_sum.reduction << points



    res = partial_sum.reduction.data()
    arithmetic_progression = np.array([upper*(upper+1)/2 for upper in arr])
    print("Expected")
    print(arithmetic_progression)
    print("Results from C++")
    print(res)
    assert np.allclose(arithmetic_progression, res)

if __name__ == "__main__":
    test_partial_sum()

