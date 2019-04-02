#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check PointsToHist adapter class
"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *

# Create the matrix
def test_points_to_hist_01():
    mat = N.arange(1, 13, dtype='d')

    # Create transformations
    points = C.Points(mat)
    adapter = C.PointsToHist(points.points, 0.)
    hist_edges = N.array(adapter.adapter.hist.datatype().hist().edges())

    # Add zero as initial bin edge
    orig = N.concatenate((N.zeros(1), mat), axis=0)

    assert (orig==hist_edges).all(), "C++ and Python results doesn't match"

# Create the matrix
def test_points_to_hist_02():
    mat = N.arange(1, 13, dtype='d')

    # Create transformations
    points = C.Points(mat)
    adapter = C.PointsToHist(points.points)
    hist_edges = N.array(adapter.adapter.hist.datatype().hist().edges())

    assert (mat==hist_edges).all(), "C++ and Python results doesn't match"

if __name__ == "__main__":
    run_unittests(globals())

