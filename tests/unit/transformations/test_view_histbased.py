#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import env

# @floatcopy(globals())
def test_view_01():
    start = 2.0
    size = 12
    edges = N.arange(start, start+12, dtype='d')
    arr_centers = N.arange(-2, -2+size-1, dtype='d')
    arr_edges   = N.arange(-1, -1+size, dtype='d')
    print('Input edges', edges)
    print('Input arr (centers)', arr_centers)
    print('Input arr (edges)', arr_edges)

    hist = C.Histogram(edges)

    hist_in = C.Histogram(edges, arr_centers)
    points_in = C.Points(arr_centers)
    points_e_in = C.Points(arr_edges)

    lims = [ (-1.0, 2.0), (0, 3), (0, 12), (1, 3), (6, 6.1), (1, 6), (10, 11),
             (0.1, 1.5), (1.1, 1.8), (5.5, 10.9), (10.1, 10.9), (10.1, 13.0) ]
    for (threshold, ceiling) in lims:
        print('Limits0 ', threshold, ceiling)
        threshold+=start
        ceiling+=start
        print('    Limits', threshold, ceiling)
        view = C.ViewHistBased(hist, threshold, ceiling)

        hist_out = view.add_input(hist_in)
        points_out = view.add_input(points_in)
        points_e_out = view.add_input(points_e_in)

        idx_thresh = N.max((int(N.floor(threshold-start)), 0))
        idx_ceiling = int(N.ceil(ceiling-start))
        res_expected = edges[idx_thresh:idx_ceiling+1]
        print('    Indices', idx_thresh, idx_ceiling)

        hist_out_res = hist_out.data()
        hist_out_exp = arr_centers[idx_thresh:idx_ceiling]
        points_out_res = points_out.data()
        points_out_exp = arr_centers[idx_thresh:idx_ceiling]
        points_e_out_res = points_e_out.data()
        points_e_out_exp = arr_edges[idx_thresh:idx_ceiling+1]

        res = N.array(view.view.view.datatype().edges)
        print('    Edges result', res)
        print('    Edges expect', res_expected)
        print('    Hist result', hist_out_res)
        print('    Hist expect', hist_out_exp)
        print('    Poits result', points_out_res)
        print('    Poits expect', points_out_exp)
        print('    Poits (edges) result', points_e_out_res)
        print('    Poits (edges) expect', points_e_out_exp)
        print()
        assert (res==res_expected).all()
        assert (hist_out_res==hist_out_exp).all()
        assert (points_out_res==points_out_exp).all()
        assert (points_e_out_res==points_e_out_exp).all()

if __name__ == "__main__":
    run_unittests(globals())

