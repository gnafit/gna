#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import env
from gna import context

# @floatcopy(globals()) # uncomment after porting the histogram
def test_histedges_linear_01():
    k, b = 1.5, 0.5
    size = 10
    edges0 = N.arange(size, dtype='d')
    edges1 = k*edges0+b

    hist_in  = C.Histogram(edges0)
    hist_out = C.HistEdgesLinear(hist_in, k, b);

    out0 = N.array(hist_in.hist.hist.datatype().edges)
    out1 = N.array(hist_out.histedges.hist.datatype().edges)

    print('    Edges0 (expected)', edges0)
    print('    Edges0', out0)
    print('    Edges1 (expected)', edges1)
    print('    Edges1', out1)

    assert (edges0-out0 == 0.0).all()
    assert (edges1-out1 == 0.0).all()

if __name__ == "__main__":
    run_unittests(globals())

