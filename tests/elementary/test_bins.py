#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class

and test the matrix memory ordering"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
import gna.constructors as C
from gna.bindings import DataType

def test_bins():
    edges = N.logspace(0, 4, 20, base=2)
    bins = C.Bins(edges)

    def cmp(text, a, b):
        good = N.allclose(a,b)
        assert good, ''.join(('\033[32mFAIL!\033[0m', a, b))
        if good:
            print(text, '\033[32mOK\033[0m', a)
        else:
            print(text, '\033[32mFAIL!\033[0m', a, b)
        print('')

    tr=bins.bins
    cmp('edges', tr.edges.data(), edges)
    cmp('centers', tr.centers.data(), (edges[:-1]+edges[1:])*0.5)
    cmp('widths', tr.widths.data(), (edges[1:]-edges[:-1]))

if __name__ == "__main__":
    test_bins()
