#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig, plot_hist
import pytest
from gna.unittest import allure_attach_file, savegraph
import os
from gna.env import env
from gna.constructors import Points, Histogram, Histogram2d
from scipy.interpolate import interp1d

from load import ROOT as R
from gna import constructors as C

integrators_2d = [
        'gl2',
        'rect2_left', 'rect2', 'rect2_right',
        'gl21',
        'rect21_left', 'rect21', 'rect21_right',
        ]
integrators_2d_margins = {
        'linear': {
            'gl2':          1.e-14,
            'rect2':        0.0004,
            'rect2_left':   0.07,
            'rect2_right':  0.06,
            'gl21':         1.e-15,
            'rect21':       0.0004,
            'rect21_left':  0.06,
            'rect21_right': 0.06,
            },
        'log': {
            'gl2':          1.e-14,
            'rect2':        0.04,
            'rect2_left':   0.15,
            'rect2_right':  0.16,
            'gl21':         1.e-14,
            'rect21':       0.0004,
            'rect21_left':  0.06,
            'rect21_right': 0.06,
            }
        }

@pytest.mark.parametrize('input_edges', [False, True])
@pytest.mark.parametrize('test_zeros', [False, [1, 3]])
@pytest.mark.parametrize('orders', [(5, 6), ([5], [6]), ([5, 10], [6, 11]), [40, 40]])
@pytest.mark.parametrize('edges', ['linear', 'log'])
@pytest.mark.parametrize('integrator_type', integrators_2d)
def test_integrator_2d(integrator_type, edges, orders, test_zeros, input_edges, tmp_path):
    xorders, yorders = orders

    if edges=='linear':
        xedges, yedges = np.linspace(0.0, 7.0, 8), np.linspace(1.0, 11.0, 6)
    elif edges=='log':
        xedges, yedges = np.geomspace(0.1, 7.0, 8), np.geomspace(1.0, 11.0, 6)
    else:
        raise AssertionError()

    xnbins = xedges.size-1
    if isinstance(xorders, list):
        if len(xorders)==1:
            xorders = np.full(xnbins, xorders[0], dtype='i')
        else:
            ointerp = interp1d(range(len(xorders)), xorders, kind='linear')
            xorders = ointerp(xedges[:-1]/xedges[-2]).astype('i')

    ynbins = yedges.size-1
    if isinstance(yorders, list):
        if len(yorders)==1:
            yorders = np.full(ynbins, yorders[0], dtype='i')
        else:
            ointerp = interp1d(range(len(yorders)), yorders, kind='linear')
            yorders = ointerp(yedges[:-1]/yedges[-2]).astype('i')


    #21d mode
    mode21 = '21' in integrator_type

    if mode21:
        yedges = yedges[:2]
        try:
            yorders = int(yorders[0])
        except:
            pass

    if test_zeros:
        if isinstance(xorders, int) or isinstance(yorders, int):
            return
        for i in test_zeros:
            xorders[i]=0
            yorders[i+1]=0

    print()
    print('Integrator: ', integrator_type)
    print('X edges: ', xedges.size, xedges)
    print('Y edges: ', yedges.size, yedges)
    try:
        print('X orders: ', xorders.size, xorders)
    except:
        print('X orders: ', xorders)
    try:
        print('Y orders: ', yorders.size, yorders)
    except:
        print('Y orders: ', yorders)

    #
    # Create 1d integrator (sample points) for given edges and integration order
    #
    integrators = dict(
            gl2=R.Integrator2GL,
            rect2=R.Integrator2Rect,
            gl21=R.Integrator21GL,
            rect21=R.Integrator21Rect
            )
    IntegratorClass = integrators[ integrator_type.split('_', 1)[0] ]
    if '_' in integrator_type: iargs = integrator_type.rsplit('_', 1)[-1],
    else:                      iargs = ()

    if not mode21:
        if input_edges:
            edges_in = Histogram2d(xedges, yedges, xedges[:-1,None]*yedges[:-1])
            integrator = IntegratorClass(xedges.size-1, xorders, yedges.size-1, yorders, *iargs)
            integrator.points.edges(edges_in)
        else:
            integrator = IntegratorClass(xedges.size-1, xorders, xedges, yedges.size-1, yorders, yedges, *iargs)
    else:
        if yedges.size!=2:
            raise Exception('In GL21 mode there should be only one bin over second axis')
        if input_edges:
            xedges_in = Histogram(xedges, xedges[:-1])
            integrator = IntegratorClass(xedges.size-1, xorders, R.nullptr, yorders, yedges[0], yedges[1], *iargs)
            integrator.points.edges(xedges_in)
        else:
            integrator = IntegratorClass(xedges.size-1, xorders, xedges, yorders, yedges[0], yedges[1], *iargs)

    integrator.points.setLabel('Integrator inputs')
    integrator.points.y.setLabel('X (points)')
    integrator.points.xedges.setLabel('X (edges)')
    integrator.points.y.setLabel('Y (points)')
    if not mode21:
        integrator.points.yedges.setLabel('Y (edges)')
    integrator.hist.setLabel('Integrator (histogram)')

    xedges = integrator.points.xedges.data()
    xwidths = xedges[1:]-xedges[:-1]
    ywidths = yedges[1:]-yedges[:-1]
    areas = xwidths[:,None]*ywidths

    xmesh = integrator.points.xmesh.data()
    ymesh = integrator.points.ymesh.data()

    #
    # Make a function
    #
    fcn_a, fcn_b = 1.0, 2.0
    def fcn(x, y):
        return np.sin(fcn_a*x+fcn_b*y)

    def fcn_int(x1, x2, y1, y2):
        weight = -1.0/(fcn_a*fcn_b)

        s1 = np.sin( fcn_a*x2 + fcn_b*y2 )
        s2 = np.sin( fcn_a*x1 + fcn_b*y1 )
        s3 = np.sin( fcn_a*x2 + fcn_b*y1 )
        s4 = np.sin( fcn_a*x1 + fcn_b*y2 )

        return weight*(s1+s2-s3-s4)

    def integr(x, y):
        x, y = np.meshgrid(x, y, indexing='ij')
        x1 = x[:-1,:-1]
        x2 = x[1: ,1:]
        y1 = y[:-1,:-1]
        y2 = y[1:,1: ]

        return fcn_int(x1, x2, y1, y2)

    fcn_values = fcn(xmesh, ymesh)
    integrals  = integr(xedges, yedges)

    fcn_o = Points(fcn_values)
    fcn_output=fcn_o.single()
    fcn_output >> integrator.hist.f
    hist_output = integrator.hist.hist
    hist_output.setLabel('output histogram')
    hist_data = hist_output.data()

    #
    # Self test
    #
    from scipy.integrate import dblquad
    ix, iy = 4, min(yedges.size-2, 2)
    x1, x2 = xedges[ix:ix+2]
    y1, y2 = yedges[iy:iy+2]

    int_s  = dblquad(lambda y, x: fcn(x, y), x1, x2, y1, y2)[0]
    int_a1 = integr( [x1, x2], [y1, y2] )[0,0]
    int_a2 = integrals[ix, iy]

    if test_zeros:
        for i in test_zeros:
            integrals[i,:]=0
            integrals[:,i+1]=0

    print('Integration self check')
    print( 'a, b', fcn_a, fcn_b )
    print( 'x', x1, x2 )
    print( 'y', y1, y2 )
    print( 'Scipy:', int_s)
    print( 'Analytic:', int_a1, int_a2 )
    print( 'Diff (scipy-analytic):', int_s-int_a1 )
    print( 'Diff (analytic):', int_a1-int_a2 )

    print()
    print('Integration check')
    print('Analytic integrals')
    print( integrals )
    print('Numeric integrals')
    print( hist_data )
    if mode21:
        diffs = hist_data-integrals.T[0]
    else:
        diffs = hist_data-integrals
    print('Diffs', np.max(np.fabs(diffs)))
    print(diffs)

    integrator.dump()

    margin = integrators_2d_margins[edges].get(integrator_type, 1.e-20)
    if margin and orders==[40, 40]:
        if mode21:
            OK = np.allclose(integrals.T[0], hist_data, rtol=0, atol=margin)
        else:
            OK = np.allclose(integrals, hist_data, rtol=0, atol=margin)
        if not OK:
            plt.show()
            assert False

    #
    # Plot data
    #
    if not mode21:
        filename = str(tmp_path/f'integrator_2d_{integrator_type}.png')

        fig = plt.figure()
        ax = plt.subplot(111, xlabel='', ylabel='', title='Integral')
        ax.minorticks_on()
        ax.set_xlim(xedges[0]-xwidths[0]*0.5, xedges[-1]+xwidths[-1]*0.5)
        ax.set_ylim(yedges[0]-ywidths[0]*0.5, yedges[-1]+ywidths[-1]*0.5)

        hist_output.plot_pcolormesh(colorbar=True, mask=0.0)
        ax.plot(xmesh.ravel(), ymesh.ravel(), 'o', color='red', markerfacecolor='none', markersize=1.0)

        paths = savefig(filename, suffix='_2d', dpi=300)
        allure_attach_file(paths[0])

        fig = plt.figure()
        ax = plt.subplot(111, xlabel='', ylabel='', title='Integral (scaled by area)', projection='3d' )
        ax.minorticks_on()
        ax.grid()

        hist_data/=areas
        hist_output.plot_bar3d(alpha=0.4, color='red')
        ax.plot_wireframe(xmesh, ymesh, fcn_values)

        paths = savefig(filename, suffix='_3d', dpi=300)
        allure_attach_file(paths[0])

        plt.close('all')

    try:
        from gna.graphviz import GNADot
        filename = str(tmp_path/f'integrator_2d_{integrator_type}_graph')
        graph = GNADot(integrator.hist)
        graph.write(f'{filename}.dot')
        graph.write(f'{filename}.pdf')
        print('Write output to:', f'{filename}.pdf')
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

