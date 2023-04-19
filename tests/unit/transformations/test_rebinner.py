#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from mpl_tools.helpers import savefig, plot_hist
import gna.constructors as C
import os
from gna.unittest import allure_attach_file
import pytest

@pytest.mark.parametrize('mode', ('static', 'inputstatic', 'inputdynamic'))
def test_rebinner(mode, tmp_path):
    edges   = N.array( [ 0.0, 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.0 ], dtype='d' )
    edges_m = N.array( [      0.1, 1.2,      3.4, 4.5,           7.8      ], dtype='d' )

    histin = C.Histogram(edges, N.ones( edges.size-1 ) )

    matrixavailable = False
    pointsin, matrixt, histout = None, None, None
    if mode=='static':
        rebin = R.Rebin(edges_m.size, edges_m, 3)
        histin >> rebin.rebin.histin
    else:
        matrixavailable = True
        if mode=='inputstatic':
            rebin = R.RebinInput(3, R.GNA.DataMutability.Static, R.GNA.DataPropagation.Propagate)
            histin >> rebin.matrix.EdgesIn
        elif mode=='inputdynamic':
            rebin = R.RebinInput(3, R.GNA.DataMutability.Dynamic, R.GNA.DataPropagation.Propagate)
            pointsin = C.Points(edges)
            pointsin >> rebin.matrix.EdgesIn
        else:
            assert False

        histout = C.Histogram(edges_m)
        histout.hist.hist >> rebin.matrix.HistEdgesOut

        histin >> rebin.rebin.histin

        matrixt = rebin.matrix

    rebin.printtransformations()
    olddata = histin.data()
    newdata = rebin.rebin.histout.data()

    assert matrixt is None or not matrixt.tainted()
    assert not rebin.rebin.tainted()

    histin.hist.taint()
    assert matrixt is None or not matrixt.tainted()
    assert rebin.rebin.tainted()
    rebin.rebin.touch()

    if histout:
        histout.hist.taint()
        assert not matrixt.tainted()
        assert not rebin.rebin.tainted()

    if pointsin:
        pointsin.points.taint()
        assert matrixt.tainted()
        assert rebin.rebin.tainted()

    rebin.rebin.touch()

    plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    # ax.grid()
    ax.set_xlabel( 'X axis' )
    ax.set_ylabel( 'Y axis' )
    ax.set_title( f'Rebinner ({mode})' )

    ax.vlines( edges, 0.0, 4.0, linestyle='--', linewidth=0.5 )
    plot_hist( edges, olddata, label='before' )
    plot_hist( edges_m, newdata, label='after' )

    ax.legend( loc='upper left' )

    path = os.path.join(str(tmp_path), 'rebinner_hist.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    if matrixavailable:
        mat = rebin.matrix.FakeMatrix.data()
        prj = mat.sum(axis=0)
        status = ((prj==1.0) + (prj==0.0)).all()
        print(status and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m')
        assert status, "Not matching results after rebin"

        mat_masked = N.ma.array(mat, mask=mat==0.0)

        #
        # Plot matrix
        #
        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( 'Source bins' )
        ax.set_ylabel( 'Target bins' )
        ax.set_title( f'Rebinning matrix ({mode})' )

        ax.matshow( mat_masked, extent=[edges[0], edges[-1], edges_m[-1], edges_m[0]] )

        path = os.path.join(str(tmp_path), 'rebinner_matrix.png')
        savefig(path, dpi=300)
        allure_attach_file(path)

        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( 'Source bins' )
        ax.set_ylabel( 'Target bins' )
        ax.set_title( f'Rebinning matrix ({mode})' )

        ax.set_ylim(edges[-1], edges[0])

        ax.vlines( edges, edges_m[0], edges_m[-1], linestyle='--', linewidth=0.5, color='gray' )
        ax.hlines( edges_m, edges[0], edges[-1], linestyle='--', linewidth=0.5, color='gray' )

        ax.pcolorfast(edges, edges_m, mat_masked)

        path = os.path.join(str(tmp_path), 'rebinner_matrix_1.png')
        savefig(path, dpi=300)
        allure_attach_file(path)
    # plt.show()
    plt.close('all')

# def test_rebinner_exception(tmp_path):
    # edges   = N.array( [ 0.0, 0.1,  1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.0 ], dtype='d' )
    # edges_m = N.array( [      0.11, 1.2,      3.4, 4.5,           7.8      ], dtype='d' )

    # histin = C.Histogram(edges, N.ones( edges.size-1 ) )
    # rebin = R.Rebin( edges_m.size, edges_m, 3 )
    # try:
        # histin >> rebin.rebin.histin
    # except Exception as e:
        # import IPython; IPython.embed()
