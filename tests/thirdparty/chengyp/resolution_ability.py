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
                ("normalizationEnergy",   (2.505, 'fixed'))
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
            normalizationEnergy = '60Co total gamma energy, MeV'
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
    evis_edges_full_input = N.arange(0.0, 12.0+1.e-6, 0.025)
    evis_edges_full_hist = C.Histogram(evis_edges_full_input, labels='Evis bin edges')
    evis_edges_full_hist >> quench.context.inputs.evis_edges_hist['00']

    #
    # Oscprob
    #
    baselinename='L'
    ns = env.ns("oscprob")
    import gna.parameters.oscillation
    gna.parameters.oscillation.reqparameters(ns)
    ns.defparameter(baselinename, central=52.0, fixed=True, label='Baseline, km')

    # Define energy range
    enu_input = N.arange(1.0, 10.0, 0.01)
    enu = C.Points(enu_input, labels='Neutrino energy, MeV')

    # Initialize oscillation variables
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

    psur = op_sum.single().data()
    from scipy.signal import argrelmin, argrelmax
    psur_minima, = argrelmin(psur)
    psur_maxima, = argrelmax(psur)
    psur_extrema = N.sort(N.concatenate( [psur_minima, psur_maxima] ))

    extrema_pos = enu_input[psur_extrema]
    print(psur_extrema)
    print(extrema_pos)
    dist_x = (extrema_pos[1:] + extrema_pos[:-1])*0.5
    dist   =  extrema_pos[1:] - extrema_pos[:-1]
    print(dist_x)
    print(dist)

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

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Enu, MeV' )
    ax.set_ylabel( 'Psur' )
    ax.set_title( 'Survival probability' )
    op_sum.single().plot_vs(enu.single(), label='full')
    ax.plot(enu_input[psur_minima], psur[psur_minima], 'o', markerfacecolor='none', label='minima')
    ax.plot(enu_input[psur_maxima], psur[psur_maxima], 'o', markerfacecolor='none', label='maxima')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Enu, MeV' )
    ax.set_ylabel( 'Dist, MeV' )
    ax.set_title( 'Extrema distance' )

    ax.plot( dist_x, dist, 'o-', markerfacecolor='none' )

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
    savefig(opts.output, suffix='_total_relative')

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
