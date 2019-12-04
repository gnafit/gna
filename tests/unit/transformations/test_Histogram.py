#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class (constructed from ROOT objects)"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
import gna.constructors as C
from gna import context
from mpl_toolkits.mplot3d import Axes3D
from mpl_tools import bindings
from mpl_tools.helpers import savefig
import os
from gna.unittest import *

def test_histogram_v01_1d(tmp_path):
    edges = np.logspace(-3, 3, 40.0)
    data  = np.arange(1.0, edges.size, dtype='d')
    hist = C.Histogram(edges, data)

    res = hist.hist.hist()
    edges_dt = np.array(hist.hist.hist.datatype().edges)

    # Plot
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('X label, log scale')
    ax.set_ylabel('entries')
    ax.set_title('Example histogram')
    ax.set_xscale('log')

    hist.hist.hist.plot_hist(label='label')
    ax.legend()

    suffix = 'histogram1d'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph(hist.hist, path)
    allure_attach_file(path)

    # Test consistency
    assert np.all(res==data)
    assert np.all(edges==edges_dt)

def test_histogram_v02_2d(tmp_path):
    edgesx = np.logspace(0, 3, 6.0, base=2.0)
    edgesy = np.linspace(0, 10, 20.0)
    data  = np.arange(1.0, (edgesx.size-1)*(edgesy.size-1)+1, dtype='d').reshape(edgesx.size-1, edgesy.size-1)

    hist = C.Histogram2d(edgesx, edgesy, data)
    res = hist.hist.hist()

    edgesx_dt = np.array(hist.hist.hist.datatype().edgesNd[0])
    edgesy_dt = np.array(hist.hist.hist.datatype().edgesNd[1])

    # Plot
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('X (column), log scale')
    ax.set_ylabel('Y row')
    ax.set_title('2d histogram example')
    ax.set_xscale('log')

    hist.hist.hist.plot_pcolor(colorbar=True)

    suffix = 'histogram2d'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    fig = plt.figure()
    ax = plt.subplot( 111, projection='3d' )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('X (column)')
    ax.set_ylabel('Y (row)')
    ax.set_title('2d histogram example (3d)')
    ax.azim-=70

    hist.hist.hist.plot_bar3d(cmap=True, colorbar=True)

    suffix = 'histogram2d_3d'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph(hist.hist, path)
    allure_attach_file(path)

    plt.show()

    # Test consistency
    assert np.all(res==data)
    assert np.all(edgesx==edgesx_dt)
    assert np.all(edgesy==edgesy_dt)

if __name__ == "__main__":
    run_unittests(globals())

