#!/usr/bin/env python

from load import ROOT as R
import numpy as np
from gna import constructors as C
from gna.bindings import common
from matplotlib import pyplot as plt
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
        stopping_power='data/data_juno/energy_model/2019_birks_cherenkov_v01/stoppingpower.txt',
        annihilation_electrons=dict(
            file='data/data_juno/energy_model/2019_birks_cherenkov_v01/hgamma2e.root',
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
                # ("normalizationEnergy",   (12.0, 'fixed'))
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
            # normalizationEnergy = 'Pessimistic norm point'
            ),
        )

    ns = env.globalns('energy')
    quench = execute_bundle(cfg, namespace=ns)
    ns.printparameters(labels=True)
    print()
    normE = ns['normalizationEnergy'].value()
    Npescint=ns['Npescint']
    kC=ns['kC']

    #
    # Input bins
    #
    evis_edges_full_input = np.arange(0.0, 12.0+1.e-6, 0.001)
    evis_edges_full_hist = C.Histogram(evis_edges_full_input, labels='Evis bin edges')
    evis_edges_full_hist >> quench.context.inputs.evis_edges_hist['00']

    #
    # HistNonLinearity transformation
    #
    reference_histogram1_input = np.zeros(evis_edges_full_input.size-1)
    reference_histogram2_input = reference_histogram1_input.copy()
    reference_histogram1_input+=1.0
    # reference_histogram2_input[[10, 20, 50, 100, 200, 300, 400]]=1.0
    reference_histogram2_input[[10, 20]]=1.0
    reference_histogram1 = C.Histogram(evis_edges_full_input, reference_histogram1_input, labels='Reference hist 1')
    reference_histogram2 = C.Histogram(evis_edges_full_input, reference_histogram2_input, labels='Reference hist 2')

    reference_histogram1 >> quench.context.inputs.lsnl.R1.values()
    reference_histogram2 >> quench.context.inputs.lsnl.R2.values()
    reference_smeared1 = quench.context.outputs.lsnl.R1
    reference_smeared2 = quench.context.outputs.lsnl.R2

    #
    # Plots and tests
    #
    savefig_old=savefig
    if opts.output and opts.output.endswith('.pdf'):
        pdfpages = PdfPages(opts.output)
        pdfpagesfilename=opts.output
        pdf=pdfpages.__enter__()
    else:
        pdf = None
        pdfpagesfilename = ''
        pdfpages = None
    def savefig(*args, **kwargs):
        close = kwargs.pop('close', False)
        if opts.individual and args and args[0]:
            savefig_old(*args, **kwargs)
        if pdf:
            pdf.savefig()

        if close:
            plt.close()


    birks_emin = quench.birks_e_p.single().data()[0]
    lsnl_e = quench.histoffset.histedges.points_truncated.data()
    lsnl_y = quench.positron_model_relative.data()
    lsnl_min_i = lsnl_y.argmin()

    #
    # Plots and tests
    #
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'dE/dx' )
    ax.set_title( 'Stopping power' )

    quench.birks_quenching_p.points.points.plot_vs(quench.birks_e_p.points.points, '-', markerfacecolor='none', markersize=2.0, label='input')
    ax.axvline(birks_emin, ls='--', label='Extrapolation boundary')
    ax.axvline(lsnl_e[lsnl_min_i]-quench.doubleme, ls='--', label='minimum location $-2m_e$', linewidth=1, color='red')
    ax.legend(loc='upper right')
    savefig(opts.output, suffix='_spower')

    ax.set_xlim(0., 2.0)
    savefig(opts.output, suffix='_spower_zoom')

    ax.set_xlim(left=0.001)
    ax.set_xscale('log')
    savefig(opts.output, suffix='_spower_log', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( "Birk's integrand" )

    quench.birks_integrand_raw.polyratio.ratio.plot_vs(quench.birks_e_p.points.points, '-o', alpha=0.5, markerfacecolor='none', markersize=2.0, label='raw')
    # birks_integrand_interpolated.plot_vs(integrator_ekin.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    lines=quench.birks_integrand_interpolated.plot_vs(quench.integrator_ekin.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    quench.birks_integrand_interpolated.plot_vs(quench.integrator_ekin.points.x, 'o', alpha=0.5, color='black', markersize=0.6)
    ax.axvline(birks_emin, ls='--', label='Extrapolation boundary')
    ax.axvline(lsnl_e[lsnl_min_i]-quench.doubleme, ls='--', label='minimum location $-2m_e$', linewidth=1, color='red')
    ax.legend(loc='lower right')

    savefig()

    ax.set_xlim(left=0.0001)
    ax.set_ylim(bottom=0.3)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    savefig(opts.output, suffix='_spower_int', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( "Birk's integral (bin by bin)" )

    quench.birks_integral.plot_hist()
    ax.axvline(birks_emin, ls='--', label='Extrapolation boundary')
    ax.axvline(lsnl_e[lsnl_min_i]-quench.doubleme, ls='--', label='minimum location $-2m_e$', linewidth=1, color='red')
    ax.legend(loc='lower right')

    savefig()

    ax.set_xlim(left=0.0001)
    ax.set_ylim(bottom=0.0)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(auto=True)
    savefig(opts.output, suffix='_spower_intc', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Electron energy (Birks)' )
    #  birks_accumulator.reduction.out.plot_vs(integrator_evis.transformations.hist, '-', markerfacecolor='none', markersize=2.0, label='partial sum')
    quench.birks_accumulator.reduction.plot_vs(quench.histoffset.histedges.points_offset)
    ax.plot([0.0, 12.0], [0.0, 12.0], '--', alpha=0.5)
    ax.axvline(lsnl_e[lsnl_min_i]-quench.doubleme, ls='--', label='minimum location $-2m_e$', linewidth=1, color='red')
    ax.legend()

    savefig(opts.output, suffix='_birks_evis')

    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.1, 0.2)
    savefig(opts.output, suffix='_birks_evis_zoom', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis/Edep' )
    ax.set_title( 'Electron energy (Birks)' )
    #  birks_accumulator.reduction.out.plot_vs(integrator_evis.transformations.hist, '-', markerfacecolor='none', markersize=2.0, label='partial sum')
    quench.histoffset.histedges.points_offset.vs_plot(quench.birks_accumulator.reduction.data()/quench.histoffset.histedges.points_offset.data())
    ax.set_ylim(0.40, 1.0)

    savefig(opts.output, suffix='_birks_evis_rel')

    ax.set_xlim(1.e-3, 2.0)
    ax.set_xscale('log')
    savefig(close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Cherenkov photons' )
    quench.cherenkov.cherenkov.ch_npe.plot_vs(quench.histoffset.histedges.points_offset)
    ax.axvline(lsnl_e[lsnl_min_i]-quench.doubleme, ls='--', label='minimum location $-2m_e$', linewidth=1, color='red')
    ax.legend()

    savefig(opts.output, suffix='_cherenkov_npe')

    ax.set_ylim(bottom=0.1)
    ax.set_yscale('log')
    savefig()

    ax.set_xlim(0.0, 2.0)
    # ax.set_ylim(0.0, 200.0)
    savefig(close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Electron model' )
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_offset)

    savefig(opts.output, suffix='_electron', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Electron model (low energy view)' )
    annihilation_gamma_evis = quench.npe_positron_offset.normconvolution.result.data()[0]
    label = 'Annihilation contribution=%.2f Npe'%annihilation_gamma_evis
    quench.electron_model_lowe.plot_vs(quench.ekin_edges_lowe, 'o', markerfacecolor='none', label='data')
    quench.electron_model_lowe_interpolated.single().plot_vs(quench.annihilation_electrons_centers.single(), '-', label='interpolation\n%s'%label)

    ax.legend(loc='upper left')

    savefig(opts.output, suffix='_electron_lowe', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Total Npe' )
    quench.positron_model.sum.outputs[0].plot_vs(quench.histoffset.histedges.points_truncated)

    savefig(opts.output, suffix='_total_npe')

    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(1000, 1500)
    savefig(opts.output, suffix='_total_npe_zoom', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Energy models comparison' )
    kC.push(0)
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_offset, label='Electron: Birks')
    kC.pop()
    Npescint.push(0)
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_offset, label='Electron: Cherenkov')
    Npescint.pop()
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_offset, label='Electron: total')
    ls = quench.positron_model.plot_vs(quench.histoffset.histedges.points_offset, label=f'Positron: total (electron + {annihilation_gamma_evis:.2f} Npe) vs Ekin', linestyle='--')
    quench.positron_model.plot_vs(quench.histoffset.histedges.points_truncated, label=f'Positron: total (electron + {annihilation_gamma_evis:.2f} Npe)', color=ls[0].get_color())
    ax.axvline(quench.doubleme, linestyle='--', label='$2m_e$')
    ax.axvline(lsnl_e[lsnl_min_i]-quench.doubleme, ls='--', label='minimum location $-2m_e$', linewidth=1, color='red')
    ax.legend(fontsize='x-small')

    savefig(opts.output, suffix='_comparison')

    ax.set_xlim(-0.05, 1.2)
    ax.set_ylim(-100, 4000)
    savefig(opts.output, suffix='_comparison_zoom', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Energy models comparison' )
    kC.push(0)
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_truncated, '-', label='Electron: Birks vs E+$2m_e$', markerfacecolor='none', markersize=2.0)
    kC.pop()
    Npescint.push(0)
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_truncated, label='Electron: Cherenkov vs vs E+$2m_e$')
    Npescint.pop()
    quench.electron_model.single().plot_vs(quench.histoffset.histedges.points_truncated, label='Electron: total vs E+$2m_e$')
    quench.positron_model.plot_vs(quench.histoffset.histedges.points_truncated, label=f'Positron: total (electron + {annihilation_gamma_evis:.2f} Npe)')
    ax.axvline(quench.doubleme, linestyle='--', label='$2m_e$')
    ax.axvline(lsnl_e[lsnl_min_i], ls='--', label='minimum location', linewidth=1, color='red')
    ax.legend(fontsize='x-small')

    savefig(opts.output, suffix='_comparison1')

    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(-50, 1500)
    savefig(opts.output, suffix='_comparison1_zoom')

    ax.set_xlim(1.01, 1.1)
    ax.set_ylim(-10, 40)
    savefig(opts.output, suffix='_comparison1_zoom1', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Positron energy model' )
    quench.positron_model_scaled.plot_vs(quench.histoffset.histedges.points_truncated, label='definition range')
    # quench.positron_model_scaled_full.plot_vs(quench.histoffset.histedges.points, '--', linewidth=1., label='full range', zorder=0.5)
    ax.plot([0.0, 12.0], [0.0, 12.0], '--', linewidth=1.0, color='black', label='equal')
    ax.vlines(normE, 0.0, normE, linestyle=':')
    ax.hlines(normE, 0.0, normE, linestyle=':')
    ax.axvline(quench.doubleme, linestyle='--', label='$2m_e$', linewidth=1)
    ax.axvline(lsnl_e[lsnl_min_i], ls='--', label='minimum', linewidth=1, color='red')
    ax.legend(loc='lower right')

    savefig(opts.output, suffix='_total')

    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(0.9, 1.2)
    savefig(opts.output, suffix='_total_zoom')

    ax.set_xlim(1.01, 1.05)
    ax.set_ylim(0.9, 0.95)

    x, y = lsnl_e[lsnl_min_i], quench.positron_model_scaled.data()[lsnl_min_i]
    step=[-0.1, 0.1]
    ax.plot(x+step, y+step, '--', color='cyan', linewidth=0.8, label='diag')
    ax.legend(loc='lower right')
    savefig(opts.output, suffix='_total_zoom1', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Positron energy model derivative' )
    lsnl_abs = quench.positron_model_scaled.data()

    widths = lsnl_e[1:]-lsnl_e[:-1]
    centers = 0.5*(lsnl_e[1:]+lsnl_e[:-1])
    diff = (lsnl_abs[1:]-lsnl_abs[:-1])/widths

    ax.plot(centers, diff, label='derivative')
    ax.plot(lsnl_e, lsnl_abs/lsnl_e, label='Evis/Edep')

    ax.axvline(quench.doubleme, linestyle='--', label='$2m_e$', linewidth=1)
    ax.axvline(lsnl_e[lsnl_min_i], ls='--', label='minimum', linewidth=1, color='red')
    ax.legend(loc='lower right')

    savefig(opts.output, suffix='_total_derivative')


    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(0.6, 1.05)
    savefig(opts.output, suffix='_total_derivative_zoom')

    ax.set_xlim(1.01, 1.1)
    ax.set_ylim(0.7, 1.)
    savefig(opts.output, suffix='_total_derivative_zoom1', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis/Edep' )
    ax.set_title( 'Positron energy nonlineairty' )
    quench.positron_model_relative.single().plot_vs(quench.histoffset.histedges.points_truncated, label='definition range')
    # quench.positron_model_relative_full.plot_vs(quench.histoffset.histedges.points, '--', linewidth=1., label='full range', zorder=0.5)
    ax.vlines(normE, 0.0, 1.0, linestyle=':')
    ax.axvline(quench.doubleme, linestyle='--', label='$2m_e$', linewidth=1)
    ax.vlines(lsnl_e[lsnl_min_i], 0, lsnl_y[lsnl_min_i], ls='--', label='minimum', linewidth=1, color='red')

    ax.legend(loc='lower right')
    ax.set_ylim(0.85, 1.1)

    savefig(opts.output, suffix='_total_relative')

    ax.set_xlim(0.75, 3)
    savefig(opts.output, suffix='_total_relative1')

    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(0.89, 0.915)
    savefig(opts.output, suffix='_total_relative2', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel('Etrue, MeV')
    ax.set_ylabel('Edep, MeV')
    ax.set_title( 'Smearing matrix' )

    quench.pm_histsmear.matrix.FakeMatrix.plot_matshow(mask=0.0, extent=[evis_edges_full_input.min(), evis_edges_full_input.max(), evis_edges_full_input.max(), evis_edges_full_input.min()], colorbar=True)
    ax.plot([0.0, 12.0], [0.0, 12.0], '--', alpha=0.5, linewidth=1.0, color='magenta')
    ax.vlines(normE, 0.0, normE, linestyle=':')
    ax.hlines(normE, 0.0, normE, linestyle=':')

    savefig(opts.output, suffix='_matrix')

    ax.set_xlim(0.8, 3.0)
    ax.set_ylim(3.0, 0.8)
    savefig(opts.output, suffix='_matrix_zoom', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Reference histogram 1' )

    reference_histogram1.single().plot_hist(linewidth=0.5, alpha=1.0, label='original')
    reference_smeared1.single().plot_hist(  linewidth=0.5, alpha=1.0, label='smeared')

    ax.legend(loc='upper right')

    savefig(opts.output, suffix='_refsmear1', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Matrix projections' )

    mat = quench.pm_histsmear.matrix.FakeMatrix.data()
    proj0 = mat.sum(axis=0)
    proj1 = mat.sum(axis=1)

    plot_hist(evis_edges_full_input, proj0, alpha=0.7, linewidth=1.0, label='Projection 0: Edep view')
    plot_hist(evis_edges_full_input, proj1, alpha=0.7, linewidth=1.0, label='Projection 1: Evis')

    ax.legend(loc='upper right')

    savefig(opts.output, suffix='_matrix_projections', close=True)

    if opts.mapping:
        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( 'E, MeV' )
        ax.set_ylabel( '' )
        ax.set_title( 'Mapping' )

        positron_model_scaled_data = quench.positron_model_scaled.single().data()
        for e1, e2 in zip(quench.histoffset.histedges.points_truncated.data(), positron_model_scaled_data):
            if e2>12.0 or e2<1.022:
                alpha = 0.05
            else:
                alpha = 0.7
            ax.plot( [e1, e2], [1.0, 0.0], '-', linewidth=2.0, alpha=alpha )
        ax.axvline(1.022, linestyle='--', linewidth=1.0)
        ax.axvline(12.0, linestyle='--', linewidth=1.0)

        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( 'E, MeV' )
        ax.set_ylabel( '' )
        ax.set_title( 'Mapping' )

        positron_model_scaled_data = quench.positron_model_scaled.single().data()
        for e1, e2 in zip(quench.histoffset.histedges.points_truncated.data(), positron_model_scaled_data):
            if e2>12.0 or e2<1.022:
                alpha = 0.05
            else:
                alpha = 0.7
            ax.plot( [e1, e2], [1.1, 0.9], '-', linewidth=2.0, alpha=alpha )

        for e1 in quench.histoffset.histedges.points.data():
            ax.axvline(e1, linestyle=':', linewidth=1.0, color='black')

        ax.axvline(1.022, linestyle='--', linewidth=1.0)
        ax.axvline(12.0, linestyle='--', linewidth=1.0)

        # ax.legend(loc='upper right')

        savefig(opts.output, suffix='_mapping_bins', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Reference histogram 2' )

    reference_histogram2.single().plot_hist(linewidth=0.5, alpha=1.0, label='original')
    reference_smeared2.single().plot_hist(  linewidth=0.5, alpha=1.0, label='smeared')
    ax.vlines(normE, 0.0, 1.0, linestyle=':')

    ax.legend(loc='upper right')

    savefig(opts.output, suffix='_refsmear2', close=True)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Entries' )
    ax.set_title( 'Annihilation gamma electrons' )

    plot_hist(quench.annihilation_electrons_edges_input, quench.annihilation_electrons_p_input)
    ax.set_yscale('log')
    savefig(opts.output, suffix='_annihilation_electrons')

    ax.set_yscale('linear')
    ax.set_xlim(0.0, 0.1)
    savefig(opts.output, suffix='_annihilation_electrons_lin', close=True)

    if pdfpages:
        pdfpages.__exit__(None,None,None)
        print('Write output figure to', pdfpagesfilename)

    savegraph(quench.histoffset.histedges.points_truncated, opts.graph, namespace=ns)

    if opts.show:
        plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-i', '--individual', help='Save individual output files', action='store_true')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('-s', '--show', action='store_true', help='Show the plots')
    parser.add_argument('-m', '--mapping', action='store_true', help='Do mapping plot')

    main( parser.parse_args() )
