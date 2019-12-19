#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class (constructed from ROOT objects)"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
import gna.constructors as C
from gna import context
from mpl_tools import bindings
from mpl_tools.helpers import savefig
import os
from gna.unittest import *

def test_histogram_v01_TH1D(tmp_path):
    rhist = R.TH1D('testhist', 'testhist', 20, -5, 5)
    rhist.FillRandom('gaus', 10000)

    hist = C.Histogram(rhist)

    buf = rhist.get_buffer()
    res = hist.hist.hist()

    # Plot
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('X label')
    ax.set_ylabel('entries')
    ax.set_title('Histogram')

    rhist.plot(alpha=0.5, linestyle='dashed', label='ROOT histogram')
    hist.hist.hist.plot_hist(alpha=0.5, linestyle='dashdot', label='GNA histogram')

    ax.legend()

    suffix = 'histogram1d'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph(hist.hist, path)
    allure_attach_file(path)
    plt.close()

    # Test consistency
    assert np.all(buf==res)

def test_histogram_v02_TH2D(tmp_path):
    rhist = R.TH2D('testhist', 'testhist', 20, 0, 10, 24, 0, 12)

    xyg=R.TF2("xyg","exp([0]*x)*exp([1]*y)", 0, 10, 0, 12)
    xyg.SetParameter(0, -1/2.)
    xyg.SetParameter(1, -1/8.)
    R.gDirectory.Add( xyg )

    rhist.FillRandom('xyg', 10000)

    hist = C.Histogram2d(rhist)

    buf = rhist.get_buffer().T
    res = hist.hist.hist()

    # Plot
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_title('ROOT histogram')

    rhist.pcolorfast(colorbar=True)

    suffix = 'histogram2d'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_title('GNA histogram')

    hist.hist.hist.plot_pcolorfast(colorbar=True)

    suffix = 'histogram2d'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph(hist.hist, path)
    allure_attach_file(path)
    plt.close()

    # Test consistency
    assert np.all(buf==res)

if __name__ == "__main__":
    run_unittests(globals())

