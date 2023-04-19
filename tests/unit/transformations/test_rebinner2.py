#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from mpl_tools.helpers import savefig, plot_hist
import gna.constructors as C
import os
from gna.unittest import allure_attach_file
import pytest

def test_rebinner2(tmp_path):
    global rebin2
    edges   = N.array( [ 0.0, 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.0 ], dtype='d' )
    edges_m = N.array( [      0.1, 1.2,      3.4, 4.5,           7.8      ], dtype='d' )

    arrayin = C.Points(N.ones( edges.size-1 ))

    pointsin, matrixt, histout = None, None, None

    rebin2 = R.Rebin2(edges.size, edges_m.size, edges, edges_m, 3)
    arrayin >> rebin2.rebin.array

    rebin2.printtransformations()
    olddata = arrayin.points.points.data()
    newdata = rebin2.rebin.histout.data()

    assert matrixt is None or not matrixt.tainted()
    assert not rebin2.rebin.tainted()

    arrayin.points.taint()
    assert matrixt is None or not matrixt.tainted()
    assert rebin2.rebin.tainted()
    rebin2.rebin.touch()

    if histout:
        histout.hist.taint()
        assert not matrixt.tainted()
        assert not rebin2.rebin.tainted()

    if pointsin:
        pointsin.points.taint()
        assert matrixt.tainted()
        assert rebin2.rebin.tainted()

    rebin2.rebin.touch()

    plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    # ax.grid()
    ax.set_xlabel( 'X axis' )
    ax.set_ylabel( 'Y axis' )
    ax.set_title( 'Rebinner2' )

    ax.vlines( edges, 0.0, 4.0, linestyle='--', linewidth=0.5 )
    plot_hist( edges, olddata, label='before' )
    plot_hist( edges_m, newdata, label='after' )

    ax.legend( loc='upper left' )

    path = os.path.join(str(tmp_path), 'rebinner2_hist.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    
    # plt.show()
    plt.close('all')

