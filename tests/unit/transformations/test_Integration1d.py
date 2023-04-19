#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig, plot_hist
import pytest
from gna.unittest import allure_attach_file, savegraph
import os
from gna.env import env
from gna.constructors import Points, Histogram, stdvector
from scipy.interpolate import interp1d

from load import ROOT as R
from gna import constructors as C

integrators_1d = [
        'rect_left', 'rect', 'rect_right',
        'rect21_left', 'rect21', 'rect21_right',
        'trap',
        'gl', 'gl21',
        ]

integrators_1d_counter = 0

integrators_1d_margins = {
        'gl':           1.e-15,
        'gl21':         1.e-15,
        'rect':         1.e-4,
        'rect21':       1.e-4,
        'trap':         2.e-4,
        'rect_left':    0.03,
        'rect_right':   0.03,
        'rect21_left':  0.03,
        'rect21_right': 0.03,
        }

@pytest.mark.parametrize('input_edges', [False, True])
@pytest.mark.parametrize('test_zeros', [False, [3, 7]])
@pytest.mark.parametrize('orders', [5, [5], [5, 10], 20])
@pytest.mark.parametrize('edges', [np.linspace(0.0, 10.0, 11), np.geomspace(1.0, 10.0, 11)])
@pytest.mark.parametrize('integrator_type', integrators_1d)
def test_integrator_1d(integrator_type, edges, orders, test_zeros, input_edges, tmp_path):
    global integrators_1d_counter

    nbins = edges.size-1
    orders_arg = orders
    if isinstance(orders, list):
        if len(orders)==1:
            orders = np.full(nbins, orders[0], dtype='i')
        else:
            ointerp = interp1d(range(len(orders)), orders, kind='linear')
            orders = ointerp(edges[:-1]/edges[-2]).astype('i')

    if test_zeros:
        if isinstance(orders, int):
            return
        for i in test_zeros:
            orders[i]=0

    print()
    print('Integrator: ', integrator_type)
    print('Edges: ', edges.size, edges)
    try:
        print('Orders: ', orders.size, orders)
    except:
        print('Orders: ', orders)

    ns = env.globalns(f'test_integrator_1d_{integrator_type}_{integrators_1d_counter}')
    integrators_1d_counter+=1
    ns.reqparameter( 'alpha', central=-0.5, sigma=0.1 )
    ns.printparameters()

    names = stdvector(('alpha',))
    with ns:
        fcn_we = R.WeightedSum(names, names)
    fcn_input=fcn_we.sum.alpha

    fcn_e = R.Exp()
    fcn_e.exp.points(fcn_we.sum.sum)
    fcn_output = fcn_e.exp.result

    # Initialize the integrator
    alpha=ns['alpha'].value()
    a, b = edges[0], edges[-1]
    integral_total_analytic = (np.exp(alpha*b) - np.exp(alpha*a))/alpha
    aa, bb = edges[:-1], edges[1:]
    integrals_analytic = (np.exp(alpha*bb) - np.exp(alpha*aa))/alpha
    if test_zeros:
        for i in test_zeros:
            integrals_analytic[i]=0
    integral_total_analytic1 = integrals_analytic.sum()
    del a, b, aa, bb

    #21d mode
    mode21 = '21' in integrator_type

    #
    # Create 1d integrator (sample points) for given edges and integration order
    #
    integrators = dict(
            gl=R.IntegratorGL,
            rect=R.IntegratorRect,
            trap=R.IntegratorTrap,
            gl21=R.Integrator21GL,
            rect21=R.Integrator21Rect
            )
    IntegratorClass = integrators[ integrator_type.split('_', 1)[0] ]
    if '_' in integrator_type: iargs1 = integrator_type.rsplit('_', 1)[-1],
    else:                 iargs1 = ()
    if mode21:            iargs2 = (1, 0.0, 1.0)
    else:                 iargs2 = ()
    iargs = iargs2 + iargs1


    if input_edges:
        edges_in = Histogram(edges, edges[:-1])
        integrator = IntegratorClass(edges.size-1, orders, R.nullptr, *iargs)
        integrator.points.edges(edges_in)
    else:
        integrator = IntegratorClass(edges.size-1, orders, edges, *iargs)

    integrator.points.setLabel('Integrator inputs')
    if mode21:
        integrator.points.y.setLabel('Y')
    integrator.hist.setLabel('Integrator (histogram)')

    edges = integrator.points.xedges.data()
    widths = edges[1:]-edges[:-1]

    #
    # Make fake gaussian data
    #
    # pass sample points as input to the function 'energy'
    if mode21:
        integrator.points.xmesh >> fcn_input
    else:
        integrator.points.x >> fcn_input
    # pass the function output to the histogram builder (integrator)
    fcn_output >> integrator.hist.f

    # read the histogram contents
    hist_output = integrator.hist.hist
    # hist_output.setLabel('output histogram')
    hist_data = hist_output.data()

    # if opts.dump:
    integrator.dump()
    print('Abscissas:', integrator.points.x.data())
    print('Widths:', widths)
    print('Centers:', integrator.points.xcenters.data())

    #
    # Calculate deviations
    #
    # Our function of interest is a guassian and should give 1 when integrated
    # Test it by summing the histogram bins
    integral_total_numeric = hist_data.sum()
    diff = integral_total_numeric-integral_total_analytic
    print('Integral (analytic)', integral_total_analytic)
    print('Integral (analytic, sum)', integrals_analytic.sum())
    print('Diff (Integral - %g):'%integral_total_analytic, diff)
    # print('Integrals (analytic)', integrals_analytic)
    # print('Integrals (calc)', hist_data)
    adiff = np.fabs(integrals_analytic-hist_data).sum()
    print('Diffs (abssum):', adiff)

    margin = integrators_1d_margins.get(integrator_type, 1.e-20)
    if margin and orders_arg==20:
        if np.fabs(diff)<margin and adiff<margin:
            pass
        else:
            assert False

        assert np.allclose(integrals_analytic, hist_data, rtol=0, atol=margin)

    if test_zeros:
        for i in test_zeros:
            assert hist_data[i]==0

    #
    # Plot data
    #
    filename = str(tmp_path/f'integrator.png')

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'x' )
    ax.set_ylabel( 'f(x)' )
    if isinstance(orders, int):
        ax.set_title( f'Integrate exponential ({integrator_type}). Diff: {diff:6g}, o>{orders}' )
    else:
        ax.set_title( f'Integrate exponential ({integrator_type}). Diff: {diff:6g}, o>{orders[0]}' )

    baropts = dict(alpha=0.5)
    x = integrator.points.x.data()
    fcnlabel = f'fcn. Int analytic: {integral_total_analytic:6f}'
    if x.size<50:
        ax.plot( x, fcn_output.data(), '-o', label=fcnlabel, markerfacecolor='none' )
    else:
        ax.plot( x, fcn_output.data(), '-', label=fcnlabel )
    hist_output.plot_hist(label=f'histogram. Int. numeric: {integral_total_numeric:6f}', **baropts)
    plot_hist( edges, hist_data/widths, label='histogram/binwidth', **baropts)
    ax.legend()

    paths=savefig(filename, dpi=300)
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

    print()

