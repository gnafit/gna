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
def test_histedges_offset_01():
    size = 13
    edges = N.arange(size, dtype='d')
    hist_in = C.Histogram(edges)
    arr = N.zeros(edges.size-1, dtype=context.current_precision_short())

    for offset in range(size-1):
        print('Offset', offset)
        view = C.HistEdgesOffset(hist_in, offset);
        edges_truncated = edges[offset:]

        trans = view.histedges

        points = trans.points.data()
        points_truncated = trans.points_truncated.data()

        hist_truncated_d = trans.hist_truncated.data()
        hist_truncated = N.array(trans.hist_truncated.datatype().edges)

        print('    Edges (expected)', edges)
        print('    Edges truncated (expected)', edges_truncated)

        print('    Points', points)
        print('    Points truncated', points_truncated)
        print('    Hist truncated', hist_truncated)
        print('    Hist truncated data', hist_truncated_d)

        assert (edges==points).all()
        assert (edges_truncated==points_truncated).all()
        assert (edges_truncated==hist_truncated).all()
        assert (0.0==hist_truncated_d).all()

# @floatcopy(globals()) # uncomment after porting the histogram
def test_histedges_offset_02():
    size = 13
    for edges_offset in (0, 1.5):
        edges = N.arange(size, dtype='d')+edges_offset
        hist_in = C.Histogram(edges)
        arr = N.zeros(edges.size-1, dtype=context.current_precision_short())

        for offset in range(size-1):
            threshold = edges[offset]+0.6
            print('Offset', offset)
            print('Threshold', threshold)
            view = C.HistEdgesOffset(hist_in, threshold);
            edges_truncated = edges[offset:]
            edges_threshold = edges_truncated.copy()
            edges_threshold[0]=threshold

            edges_offset = edges_threshold.copy()
            edges_offset-=threshold

            trans = view.histedges

            points = trans.points.data()
            points_truncated = trans.points_truncated.data()
            points_threshold = trans.points_threshold.data()
            points_offset = trans.points_offset.data()

            hist_truncated_d = trans.hist_truncated.data()
            hist_truncated = N.array(trans.hist_truncated.datatype().edges)

            hist_threshold_d = trans.hist_threshold.data()
            hist_threshold = N.array(trans.hist_threshold.datatype().edges)

            hist_offset_d = trans.hist_offset.data()
            hist_offset = N.array(trans.hist_offset.datatype().edges)

            print('    Edges (expected)', edges)
            print('    Edges truncated (expected)', edges_truncated)
            print('    Edges threshold (expected)', edges_threshold)
            print('    Edges offset (expected)', edges_offset)

            print('    Points', points)
            print('    Points truncated', points_truncated)
            print('    Points threshold', points_threshold)
            print('    Points offset', points_offset)
            print('    Hist truncated', hist_truncated)
            print('    Hist truncated data', hist_truncated_d)
            print('    Hist threshold', hist_threshold)
            print('    Hist threshold data', hist_threshold_d)
            print('    Hist offset', hist_offset)
            print('    Hist offset data', hist_offset_d)

            assert (edges==points).all()
            assert (edges_truncated==points_truncated).all()
            assert (edges_threshold==points_threshold).all()
            assert (edges_offset==points_offset).all()
            assert (edges_truncated==hist_truncated).all()
            assert (0.0==hist_truncated_d).all()
            assert (edges_threshold==hist_threshold).all()
            assert (0.0==hist_threshold_d).all()
            assert (edges_offset==hist_offset).all()
            assert (0.0==hist_offset_d).all()

if __name__ == "__main__":
    run_unittests(globals())

