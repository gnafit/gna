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

def main(args):
    global savefig
    cfg = NestedDict(
        bundle = dict(
            name='energy_nonlinearity_birks_cherenkov',
            version='v01',
            nidx=[ ('r', 'reference', ['R1', 'R2']) ],
            major=[],
            ),
        stopping_power='stoppingpower.txt',
        annihilation_electrons=('input/hgamma2e.root', 'hgamma2e_1KeV'),
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
                ("Npesc",               (1341.38, 'fixed')),
                ("kC",                      (0.5, 'fixed')),
                ("normalizationEnergy",   (2.505, 'fixed'))
             ],
            mode='relative'
            ),
        fill_matrix=True,
        labels = dict(
            normalizationEnergy = '60Co total gamma energy, MeV'
            ),
        )

    ns = env.globalns('energy')
    quench = execute_bundle(cfg, namespace=ns)
    ns.printparameters(labels=True)
    print()

    #
    # HistNonLinearity transformation
    #
    reference_histogram1_input = N.zeros(quench.epos_edges_full_input.size-1)
    reference_histogram2_input = reference_histogram1_input.copy()
    reference_histogram1_input+=1.0
    reference_histogram2_input[[10, 20, 50, 100, 200, 300, 400]]=1.0
    reference_histogram1 = C.Histogram(quench.epos_edges_full_input, reference_histogram1_input, labels='Reference hist 1')
    reference_histogram2 = C.Histogram(quench.epos_edges_full_input, reference_histogram2_input, labels='Reference hist 2')

    reference_histogram1 >> quench.context.inputs.lsnl_edges.values()
    reference_histogram1 >> quench.context.inputs.lsnl.R1.values()
    reference_histogram2 >> quench.context.inputs.lsnl.R2.values()
    reference_smeared1 = quench.context.outputs.lsnl.R1
    reference_smeared2 = quench.context.outputs.lsnl.R2

    #
    # Plots and tests
    #
    if args.output and args.output.endswith('.pdf'):
        pdfpages = PdfPages(args.output)
        savefig_old=savefig
        pdf=pdfpages.__enter__()
        def savefig(*args, **kwargs):
            if args and args[0]:
                savefig_old(*args, **kwargs)
            pdf.savefig()
    else:
        pdf = None

    #
    # Plots and tests
    #
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'dE/dx' )
    ax.set_title( 'Stopping power' )

    quench.birks_quenching_p.points.points.plot_vs(quench.birks_e_p.points.points, '-', markerfacecolor='none', markersize=2.0, label='input')
    ax.legend(loc='upper right')
    savefig(args.output, suffix='_spower')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( "Birk's integrand" )

    quench.birks_integrand_raw.polyratio.ratio.plot_vs(quench.birks_e_p.points.points, '-o', alpha=0.5, markerfacecolor='none', markersize=2.0, label='raw')
    # birks_integrand_interpolated.plot_vs(integrator_ekin.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    lines=quench.birks_integrand_interpolated.plot_vs(quench.integrator_ekin.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    quench.birks_integrand_interpolated.plot_vs(quench.integrator_ekin.points.x, 'o', alpha=0.5, color='black', markersize=0.6)
    ax.legend(loc='lower right')

    savefig()

    ax.set_xlim(left=0.0001)
    ax.set_ylim(bottom=0.5)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    savefig(args.output, suffix='_spower_int')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( "Birk's integral (bin by bin)" )

    quench.birks_integral.plot_hist()

    savefig()

    ax.set_xlim(left=0.0001)
    ax.set_ylim(bottom=0.0)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(auto=True)
    savefig(args.output, suffix='_spower_intc')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Electron energy (Birks)' )
    #  birks_accumulator.reduction.out.plot_vs(integrator_epos.transformations.hist, '-', markerfacecolor='none', markersize=2.0, label='partial sum')
    quench.birks_accumulator.reduction.plot_vs(quench.ekin_edges_p)
    ax.plot([0.0, 12.0], [0.0, 12.0], '--', alpha=0.5)

    savefig(args.output, suffix='_birks_evis')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis/Edep' )
    ax.set_title( 'Electron energy (Birks)' )
    #  birks_accumulator.reduction.out.plot_vs(integrator_epos.transformations.hist, '-', markerfacecolor='none', markersize=2.0, label='partial sum')
    quench.ekin_edges_p.vs_plot(quench.birks_accumulator.reduction.data()/quench.ekin_edges_p.data())
    ax.set_ylim(0.65, 1.0)

    savefig(args.output, suffix='_birks_evis_rel')

    ax.set_xlim(1.e-3, 2.0)
    ax.set_xscale('log')
    savefig()

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Cherenkov photons' )
    quench.cherenkov.cherenkov.ch_npe.plot_vs(quench.ekin_edges_p)

    savefig(args.output, suffix='_cherenkov_npe')

    ax.set_ylim(bottom=0.1)
    ax.set_yscale('log')
    savefig()

    ax.set_xlim(0.0, 2.0)
    # ax.set_ylim(0.0, 200.0)
    savefig()

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Electron model' )
    quench.electron_model.single().plot_vs(quench.ekin_edges_p)

    savefig(args.output, suffix='_electron')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Electron model (low energy view)' )
    annihilation_gamma_evis = quench.npe_positron_offset.normconvolution.result.data()[0]
    label = 'Annihilation contribution=%.2f Npe'%annihilation_gamma_evis
    quench.electron_model_lowe.single().plot_vs(quench.ekin_edges_lowe.view.view, 'o', markerfacecolor='none', label='data')
    quench.electron_model_lowe_interpolated.single().plot_vs(quench.annihilation_electrons_centers.single(), '-', label='{interpolation|%s}'%label)

    ax.legend(loc='upper left')

    savefig(args.output, suffix='_electron_lowe')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Total Npe' )
    quench.positron_model.sum.outputs[0].plot_vs(quench.epos_edges.single())

    savefig(args.output, suffix='_total_npe')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Positron energy model' )
    quench.positron_model_scaled.plot_vs(quench.epos_edges.single(), label='definition range')
    quench.positron_model_scaled_full.single().plot_vs(quench.epos_edges_full.single(), '--', label='full range')
    ax.legend(loc='upper left')

    savefig(args.output, suffix='_total')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis/Edep' )
    ax.set_title( 'Positron energy nonlineairty' )
    quench.positron_model_relative.single().plot_vs(quench.epos_edges.single(), label='definition range')
    quench.positron_model_relative_full.single().plot_vs(quench.epos_edges_full.single(), '--', label='full range')

    ax.legend(loc='lower right')

    savefig(args.output, suffix='_total_relative')

    ax.set_xlim(left=1.0)
    ax.set_ylim(0.5, 1.7)
    savefig(args.output, suffix='_total_relative1')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('Etrue, MeV')
    ax.set_ylabel('Edep, MeV')
    ax.set_title( 'Smearing matrix' )

    quench.pm_histsmear.matrix.FakeMatrix.plot_matshow(mask=0.0, extent=[0.0, quench.epos_edges_full_input.max(), quench.epos_edges_full_input.max(), 0.0], colorbar=True)
    ax.plot([0.0, 12.0], [0.0, 12.0], '--', alpha=0.5, linewidth=1.0)

    savefig(args.output, suffix='_matrix')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Reference histogram 1' )

    reference_histogram1.single().plot_hist(linewidth=0.5, alpha=1.0, label='original')
    reference_smeared1.single().plot_hist(  linewidth=0.5, alpha=1.0, label='smeared')

    ax.legend(loc='upper right')

    savefig(args.output, suffix='_refsmear1')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Matrix projections' )

    mat = quench.pm_histsmear.matrix.FakeMatrix.data()
    proj0 = mat.sum(axis=0)
    proj1 = mat.sum(axis=1)

    plot_hist(quench.epos_edges_full_input, proj0, alpha=0.7, linewidth=1.0, label='Projection 0: Edep view')
    plot_hist(quench.epos_edges_full_input, proj1, alpha=0.7, linewidth=1.0, label='Projection 1: Evis')

    ax.legend(loc='upper right')

    savefig(args.output, suffix='_matrix_projections')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Mapping' )

    positron_model_scaled_data = quench.positron_model_scaled.single().data()
    for e1, e2 in zip(quench.epos_edges_input, positron_model_scaled_data):
        ax.plot( [e1, e2], [1.0, 0.0], '-', linewidth=2.0, alpha=0.5 )
    ax.axvline(12.0, linestyle='--')

    # ax.legend(loc='upper right')

    savefig(args.output, suffix='_mapping')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Reference histogram 2' )

    reference_histogram2.single().plot_hist(linewidth=0.5, alpha=1.0, label='original')
    reference_smeared2.single().plot_hist(  linewidth=0.5, alpha=1.0, label='smeared')

    ax.legend(loc='upper right')

    savefig(args.output, suffix='_refsmear2')

    if pdfpages:
        pdfpages.__exit__(None,None,None)

    savegraph(quench.epos_edges.single(), args.graph, namespace=ns)

    if args.show:
        P.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('-s', '--show', action='store_true', help='Show the plots')

    main( parser.parse_args() )
