#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class (constructed from ROOT objects)"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
import gna.constructors as C
from gna import context
from gna.bindings import DataType, provided_precisions
from mpl_tools import bindings
from gna.unittest import *

def test_points_v01_TH1D():
    hist = R.TH1D('testhist', 'testhist', 10, -5, 5)
    hist.FillRandom('gaus', 10000)

    p = C.Points(hist)

    buf = hist.get_buffer()
    res = p.points.points()

    assert np.all(buf==res)

def test_points_v02_TH2D():
    hist = R.TH2D('testhist', 'testhist', 10, -5, 5, 12, -6, 6)
    hist.FillRandom('gaus', 10000)

    p = C.Points(hist)

    buf = hist.get_buffer()
    res = p.points.points()

    assert np.all(buf==res)

if __name__ == "__main__":
    run_unittests(globals())

