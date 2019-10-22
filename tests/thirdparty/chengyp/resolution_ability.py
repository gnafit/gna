#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna.bindings import common
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig, add_to_labeled_items, plot_hist
from gna.graphviz import savegraph
from gna.env import env
from matplotlib.backends.backend_pdf import PdfPages
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict

def main(opts):
    global savefig
    cfg = NestedDict(
        bundle = dict(
            name='energy_nonlinearity_birks_cherenkov',
            version='v01',
            nidx=[ ('r', 'reference', ['R1', 'R2']) ],
            major=[],
            ),
        stopping_power='stoppingpower.txt',
        annihilation_electrons=dict(
            file='input/hgamma2e.root',
            histogram='hgamma2e_1KeV',
            scale=1.0/50000 # event simulated
            ),
        pars = uncertaindict(
            [
                ('birks.Kb0',               (1.0, 'fixed')),
                ('birks.Kb1',           (15.2e-3, 0.1776)),
                # ('birks.Kb2',           (0.0, 'fixed')),
                ("cherenkov.E_0",         (0.165, 'fixed')),
                ("cherenkov.p0",  ( -7.26624e+00, 'fixed')),
                ("cherenkov.p1",   ( 1.72463e+01, 'fixed')),
                ("cherenkov.p2",  ( -2.18044e+01, 'fixed')),
                ("cherenkov.p3",   ( 1.44731e+01, 'fixed')),
                ("cherenkov.p4",   ( 3.22121e-02, 'fixed')),
                ("Npescint",            (1341.38, 0.0059)),
                ("kC",                      (0.5, 0.4737)),
                ("normalizationEnergy",   (12.0, 'fixed'))
             ],
            mode='relative'
            ),
        integration_order = 2,
        correlations_pars = [ 'birks.Kb1', 'Npescint', 'kC' ],
        correlations = [ 1.0,   0.94, -0.97,
                         0.94,  1.0,  -0.985,
                        -0.97, -0.985, 1.0   ],
        fill_matrix=True,
        labels = dict(
            normalizationEnergy = 'Pessimistic'
            ),
        )

    ns = env.globalns('energy')
    quench = execute_bundle(cfg, namespace=ns)
    ns.printparameters(labels=True)
    print()
    normE = ns['normalizationEnergy'].value()

    #
    # Input bins
    #
    evis_edges_full_input = N.arange(0.0, 15.0+1.e-6, 0.025)
    evis_edges_full_hist = C.Histogram(evis_edges_full_input, labels='Evis bin edges')
    evis_edges_full_hist >> quench.context.inputs.evis_edges_hist['00']

    #
    # Python energy model interpolation function
    #
    from scipy.interpolate import interp1d
    lsnl_x = quench.histoffset.histedges.points_truncated.data()
    lsnl_y = quench.positron_model_relative.single().data()
    lsnl_fcn = interp1d(lsnl_x, lsnl_y, kind='quadratic')

    #
    # Energy resolution
    #
    def eres_sigma_rel(edep):
        return 0.03/edep**0.5

    def eres_sigma_abs(edep):
        return 0.03*edep**0.5

    #
    # Energy offset
    #
    from physlib import pc
    edep_offset = pc.DeltaNP - pc.ElectronMass

    #
    # Oscprob
    #
    baselinename='L'
    ns = env.ns("oscprob")
    import gna.parameters.oscillation
    gna.parameters.oscillation.reqparameters(ns)
    ns.defparameter(baselinename, central=52.0, fixed=True, label='Baseline, km')

    #
    # Define energy range
    #
    enu_input = N.arange(1.8, 15.0+1.e-6, 0.001)
    edep_input = enu_input - edep_offset
    edep_lsnl = edep_input * lsnl_fcn(edep_input)

    # Initialize oscillation variables
    enu = C.Points(enu_input, labels='Neutrino energy, MeV')
    component_names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
    with ns:
        R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), component_names, ns=ns)

        labels=['Oscillation probability|%s'%s for s in ('component 12', 'component 13', 'component 23', 'full', 'probsum')]
        oscprob = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae(), baselinename, labels=labels)

    enu >> oscprob.full_osc_prob.Enu
    enu >> (oscprob.comp12.Enu, oscprob.comp13.Enu, oscprob.comp23.Enu)

    unity = C.FillLike(1, labels='Unity')
    enu >> unity.fill.inputs[0]
    with ns:
        op_sum = C.WeightedSum(component_names, [unity.fill.outputs[0], oscprob.comp12.comp12, oscprob.comp13.comp13, oscprob.comp23.comp23], labels='Oscillation probability sum')

    psur_nh = op_sum.single().data().copy()
    ns['Alpha'].set('inverted')
    psur_ih = op_sum.single().data().copy()

    def get_intersection(*arrs):
        maxlen = min(arr.size for arr in arrs)
        return (arr[-maxlen:] for arr in arrs)

    def get_extrema(arr):
        from scipy.signal import argrelmin, argrelmax
        minima, = argrelmin(arr)
        maxima, = argrelmax(arr)

        return minima, maxima

    def merge_extrema(*args):
        if args[0][0] > args[0][1]:
            args = reversed(args)

        return N.stack(args).T.ravel()

    psur_nh_minima, psur_nh_maxima = get_extrema(psur_nh)
    psur_ih_minima, psur_ih_maxima = get_extrema(psur_ih)

    psur_nh_maxima, psur_ih_maxima = get_intersection(psur_nh_maxima, psur_ih_maxima)

    def get_smaller_diff(arr, brr):
        a = arr[1:-1] - brr[0:-2]
        b = arr[1:-1] - brr[1:-1]
        c = arr[1:-1] - brr[2:]
        diff = N.fabs(N.stack((a,b,c)))

        a = arr[1:-1] + brr[0:-2]
        b = arr[1:-1] + brr[1:-1]
        c = arr[1:-1] + brr[2:]
        center = N.stack((a,b,c))*0.5

        idx = N.argmin(diff, axis=0)
        idx2 = N.indices(idx.shape)

        return center[idx, idx2][0], diff[idx, idx2][0]

    def get_smaller_diff_and_merge(arr):
        arr_max = arr[psur_nh_maxima]
        brr_max = arr[psur_ih_maxima]
        x_max, y_max = get_smaller_diff(arr_max, brr_max)

        return x_max, y_max

    data_x_enu, data_y_enu = get_smaller_diff_and_merge(enu_input)
    data_x_edep, data_y_edep = get_smaller_diff_and_merge(edep_input)
    data_x_lsnl, data_y_lsnl = get_smaller_diff_and_merge(edep_lsnl)

    #
    # Plots and tests
    #
    if opts.output and opts.output.endswith('.pdf'):
        pdfpages = PdfPages(opts.output)
        pdfpagesfilename=opts.output
        savefig_old=savefig
        pdf=pdfpages.__enter__()
        def savefig(*args, **kwargs):
            if opts.individual and args and args[0]:
                savefig_old(*args, **kwargs)
            pdf.savefig()
    else:
        pdf = None
        pdfpagesfilename = ''
        pdfpages = None

    eshift = -0.8

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis/Edep' )
    ax.set_title( 'Positron energy nonlineairty' )
    quench.positron_model_relative.single().plot_vs(quench.histoffset.histedges.points_truncated, label='definition range')
    quench.positron_model_relative_full.plot_vs(quench.histoffset.histedges.points, '--', linewidth=1., label='full range', zorder=0.5)
    ax.vlines(normE, 0.0, 1.0, linestyle=':')

    ax.legend(loc='lower right')
    ax.set_ylim(0.8, 1.05)
    savefig(opts.output, suffix='_total_relative')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( r'$\sigma/E$' )
    ax.set_title('Energy resolution')
    ax.plot(edep_input, eres_sigma_rel(edep_input), '-')

    savefig(opts.output, suffix='_eres_rel')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( r'$\sigma$' )
    ax.set_title('Energy resolution')
    ax.plot(edep_input, eres_sigma_abs(edep_input), '-')

    savefig(opts.output, suffix='_eres_abs')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Enu, MeV' )
    ax.set_ylabel( 'Psur' )
    ax.set_title( 'Survival probability' )

    color_nh = ax.plot(enu_input, psur_nh, label='NH')[0].get_color()
    color_ih = ax.plot(enu_input, psur_ih, label='IH')[0].get_color()
    ax.plot(enu_input[psur_nh_maxima], psur_nh[psur_nh_maxima], 'd', markerfacecolor='none', color=color_nh)
    ax.plot(enu_input[psur_ih_maxima], psur_ih[psur_ih_maxima], 'o', markerfacecolor='none', color=color_ih)

    ax.legend()
    savefig(opts.output, suffix=('psur', 'enu'))

    ax.set_xlim(right=5.0)
    savefig(opts.output, suffix='_psur_enu_zoom')

    ax.set_xlim(2.0, 3.0)
    ax.set_ylim(0.0, 0.7)
    savefig(opts.output, suffix='_psur_enu_zoom1')

    ax.vlines(data_x_enu, 0, 1, linestyle='-', linewidth=1.0, color='magenta', label='between maxima')
    ax.set_ylim(0.0, 0.7)
    ax.grid(False)
    ax.legend()

    savefig(opts.output, suffix=('psur', 'enu', 'lines'))

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Enu, MeV' )
    ax.set_ylabel( 'Dist, MeV' )
    ax.set_title( 'Nearest peaks distance' )

    ax.plot(data_x_enu, data_y_enu, 'o-', markerfacecolor='none' )

    savefig(opts.output, suffix='_dist_enu')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Psur' )
    ax.set_title( 'Survival probability' )

    color_nh = ax.plot(edep_input, psur_nh, '--', label='NH')[0].get_color()
    # color_ih = ax.plot(edep_input, psur_ih, '--', label='IH')[0].get_color()
    ax.plot(edep_lsnl, psur_nh, label='NH with quenching', color=color_nh)[0]
    ax.plot(edep_lsnl, psur_ih, label='IH with quenching', color=color_ih)[0]

    ax.legend(loc='lower right')
    savefig(opts.output, suffix='_psur_edep')

    # ax.set_xscale('log')
    ax.set_xlim(eshift, 5.0+eshift)
    ax.legend(loc='upper right')
    savefig(opts.output, suffix='_psur_edep_zoom')

    ax.set_xlim(2.0+eshift, 3.0+eshift)
    ax.set_ylim(0.0, 0.7)
    savefig(opts.output, suffix='_psur_edep_zoom1')

    ax.vlines(data_x_edep, 0, 1, linestyle='--', linewidth=1.0, alpha=0.5, color='teal', label='between maxima')
    ax.vlines(data_x_lsnl, 0, 1, linestyle='-', linewidth=1.0, alpha=0.8, color='magenta', label='between maxima quench.')
    ax.set_ylim(0.0, 0.7)
    ax.grid(False)

    ax.legend(loc='upper right')
    savefig(opts.output, suffix='_psur_edep_lines')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Dist, MeV' )
    ax.set_title( 'Nearest peaks distance' )

    ax.plot( data_x_edep, data_y_edep, '-', markerfacecolor='none', label='true' )
    ax.plot( data_x_lsnl, data_y_lsnl, '-', markerfacecolor='none', label='quenching' )
    ax.plot( edep_input, eres_sigma_abs(edep_input), '-', markerfacecolor='none', label=r'$\sigma$' )
    ax.set_xlim(left=1.0)

    ax.legend(loc='upper left')

    savefig(opts.output, suffix='_dist')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( r'Dist/$\sigma$' )
    ax.set_title( 'Resolution ability' )

    x1, y1 = data_x_edep, data_y_edep/eres_sigma_abs(data_x_edep)
    x2, y2 = data_x_lsnl, data_y_lsnl/eres_sigma_abs(data_x_lsnl)

    ax.plot(x1, y1, '-', markerfacecolor='none', label='true')
    ax.plot(x2, y2, '-', markerfacecolor='none', label='quenching')

    ax.legend()
    savefig(opts.output, suffix='_ability')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( r'Dist/$\sigma$' )
    ax.set_title( 'Resolution ability difference (quenching-true)' )

    y2fcn = interp1d(x2, y2, bounds_error=False, fill_value=N.nan)
    y2_on_x1 = y2fcn(x1)
    diff = y2_on_x1 - y1
    from scipy.signal import savgol_filter
    diff = savgol_filter(diff, 21, 3)

    ax.plot(x1, diff)

    savefig(opts.output, suffix='_ability_diff')

    if pdfpages:
        pdfpages.__exit__(None,None,None)
        print('Write output figure to', pdfpagesfilename)

    savegraph(quench.histoffset.histedges.points_truncated, opts.graph, namespace=ns)

    if opts.show:
        P.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-i', '--individual', help='Save individual output files', action='store_true')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('-s', '--show', action='store_true', help='Show the plots')
    parser.add_argument('-m', '--mapping', action='store_true', help='Do mapping plot')

    main( parser.parse_args() )
