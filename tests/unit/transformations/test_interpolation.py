#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig
import pytest
from gna.unittest import allure_attach_file, savegraph

from load import ROOT as R
from gna import constructors as C

@pytest.mark.parametrize('strategy', [R.GNA.Interpolation.Strategy.Extrapolate, R.GNA.Interpolation.Strategy.Constant])
@pytest.mark.parametrize('interpolator', [R.InterpLinear, R.InterpExpo, R.InterpLog, R.InterpLogx, R.InterpConst])
def test_interpolators(interpolator, strategy, tmp_path):
    '''Test various interpolators'''

    segments   = np.arange(1.0, 10.1, 1.5, dtype='d')
    segments_t = C.Points(segments)

    points   = np.stack([np.linspace(0.0+i, 12.+i, 61, dtype='d') for i in [0, -0.1, 0.1, 0.3, 0.5]]).T
    points_t = C.Points(points)

    fcn = np.exp( -(segments-segments[0])*0.5 )
    fcn = np.exp(segments**(-0.5))
    fcn_t = C.Points(fcn)

    ie = interpolator()

    ie.set_underflow_strategy(strategy)
    ie.set_overflow_strategy(strategy)

    ie.interpolate(segments_t, fcn_t, points_t)
    seg_idx = ie.insegment.insegment.data()

    res = ie.interp.interp.data()

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    strategy_name = "Constant" if strategy is R.GNA.Interpolation.Strategy.Constant else "Extrapolate"
    title = ie.__class__.__name__.strip('Interp') + ", " + "Strategy: {}".format(strategy_name)
    ax.set_title(title)

    ax.plot( segments, fcn, 'o-', markerfacecolor='none', label='coarse function', linewidth=0.5, alpha=0.5 )

    markers='os*^vh'
    for i, (p, r) in enumerate(zip(points.T, res.T)):
        ax.plot( p, r, '.', label='interpolation, col %i'%i, marker=markers[i], markersize=1.5 )
        break

    ax.legend(loc='upper right')

    suffix = '{}_{}'.format(ie.__class__.__name__, strategy_name)

    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph(points_t.points, path, verbose=False)
    allure_attach_file(path)
    plt.close()
