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

# #
# # Create the matrix
# #
# def test_points_to_hist():
    # mat = N.arange(1, 13)

    # print( mat )

    # #
    # # Create transformations
    # #
    # points = C.Points(mat)
    # adapter = C.PointsToHist(points.points)
    # adapter.adapter.hist.data()
    # import IPython
    # IPython.embed()

    # #
    # # Dump
    # #

    # #  assert N.allclose(mat, res), "C++ and Python results doesn't match"

# if __name__ == "__main__":
    # glb = globals()
    # for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        # print('call ', fcn)
        # glb[fcn]()
        # print()

    # print('All tests are OK!')

