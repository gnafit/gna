#!/usr/bin/env python

import numpy as np
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig
import pytest
from gna.unittest import allure_attach_file, savegraph

from load import ROOT as R
from gna import constructors as C

@pytest.mark.parametrize('repeat', [False, (2, 1), (2, 1.5)])
@pytest.mark.parametrize('strategy', [R.GNA.Interpolation.Strategy.Extrapolate, R.GNA.Interpolation.Strategy.Constant, R.GNA.Interpolation.Strategy.NearestEdge])
@pytest.mark.parametrize('interpolator', [R.InterpLinear, R.InterpExpo, R.InterpLog, R.InterpLogx, R.InterpConst])
def test_interpolators(interpolator, strategy, repeat, tmp_path):
    '''Test various interpolators'''

    segments   = np.arange(1.0, 10.1, 1.5, dtype='d')
    fcn = np.exp(segments**(-0.5))
    checkall = True
    if repeat is not False:
        # Check the bahaviour of the interpolator in case of the repeated point
        # and in case of a dicontinuity
        idx, scale = repeat
        segments = np.insert(segments, idx, segments[idx])
        fcn = np.insert(fcn, idx, scale*fcn[idx])
        checkall=False

    points   = np.stack([np.linspace(0.0+i, 12.+i, 61, dtype='d') for i in [0, -0.1, 0.1, 0.3, 0.5]]).T
    if interpolator==R.InterpLogx:
        points=points[1:]
    fcn_all = np.exp(points**(-0.5))

    segments_t = C.Points(segments)
    points_t = C.Points(points)
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
    strategy_name = ['Constant', 'Extrapolate', 'NearestEdge'][int(strategy)]
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

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph(points_t.points, path, verbose=False)
    allure_attach_file(path)

    plt.close('all')

    #
    # Test that there are now nans/infs
    #
    assert np.isfinite(res).all()

    #
    # Test the interpolation of the node points
    #
    node_points_idx = np.isin(points, segments)
    assert node_points_idx.any()
    # node_points = points[node_points_idx]
    node_interp = res[node_points_idx]
    node_values = fcn_all[node_points_idx]
    print('Nodes', segments)
    print('Found node points', points[node_points_idx])
    print('Fcn values:', fcn)
    print('Node values:', node_values)
    print('Node interpolation:', node_interp)
    print('Diff:', node_interp-node_values)
    print()

    fp_tolerance = np.finfo('d').resolution
    if checkall:
        if interpolator is R.InterpConst:
            a, b = node_interp[:-1], node_values[:-1]
        else:
            a, b = node_interp, node_values

        if not np.allclose(a, b, rtol=0.0, atol=fp_tolerance):
            ie.printtransformations(data=True)
            assert False

    #
    # Test outside of the segments
    #
    for i, (left, right) in enumerate(zip(segments[:-1], segments[1:])):
        mask = (points>left)*(points<right)
        points1 = points[mask]
        fcn1 = fcn_all[mask]
        res1 = res[mask]

        vmax, vmin=fcn[i], fcn[i+1]
        if vmin>vmax:
            vmin, vmax = vmax, vmin

        assert(res1<=vmax).all()
        assert(res1>=vmin).all()

    mask1 = points<segments[0]
    mask2 = points>segments[-1]
    if strategy==R.GNA.Interpolation.Strategy.Constant:
        assert (res[mask1]==0.0).all()
        assert (res[mask2]==0.0).all()
    elif strategy==R.GNA.Interpolation.Strategy.NearestEdge:
        assert np.allclose(res[mask1], fcn[0], rtol=0, atol=fp_tolerance)
        assert np.allclose(res[mask2], fcn[-1], rtol=0, atol=fp_tolerance)
    else:
        assert (res[mask1]!=0.0).all()
        assert (res[mask2]!=0.0).all()

    print()

