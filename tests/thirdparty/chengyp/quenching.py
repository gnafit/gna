#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna.bindings import common
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig, add_to_labeled_items
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.graphviz import savegraph
from gna.env import env
from matplotlib.backends.backend_pdf import PdfPages

dtype_spower = [ ('e', 'd'), ('temp1', 'd'), ('temp2', 'd'), ('dedx', 'd') ]

def load_electron_distribution(filename, objname):
    file = R.TFile(filename, 'read')
    hist = file.Get(objname)
    buf = get_buffer_hist1(hist).copy()
    buf/=buf.sum()
    edges = get_bin_edges_axis(hist.GetXaxis())
    centers = (edges[:-1] + edges[1:])*0.5

    return edges, centers, C.Points(centers, labels='Annihilation gamma E centers'), buf, C.Points(buf, labels='Annihilation gamma weights')

def main(args):
    global savefig
    ns = env.globalns('energy')
    nsb = ns('birks')
    nsb.defparameter('Kb0', central=1.0, fixed=True, label='Kb0=1')
    nsb.defparameter('Kb1', central=0.0062, fixed=True, label="Birk's 1st constant (E')")
    nsb.defparameter('Kb2', central=1.5e-6, fixed=True, label="Birk's 2nd constant (E'')")
    nsc = ns('cherenkov')
    nsc.defparameter("E_0", central=0.165, fixed=True)
    nsc.defparameter("p0",  central=-7.26624e+00, fixed=True)
    nsc.defparameter("p1",  central=1.72463e+01,  fixed=True)
    nsc.defparameter("p2",  central=-2.18044e+01, fixed=True)
    nsc.defparameter("p3",  central=1.44731e+01,  fixed=True)
    nsc.defparameter("p4",  central=3.22121e-02,  fixed=True)

    Nsc = ns.defparameter("Npesc", central=1341.38, fixed=True, label='Scintillation responce, Npe/MeV')
    kC = ns.defparameter("kC", central=1., fixed=True, label='Cerenkov contribution normalization')

    from physlib import pdg
    emass = ns.defparameter("emass", central=pdg['live']['ElectronMass'], fixed=True, label='Electron mass, MeV')
    doubleme = 2*emass.value()
    ns.defparameter("ngamma", central=2.0, fixed=True, label='Number of e+e- annihilation gammas')
    energy_scale_normalization_point = ns.defparameter('Co60gamma', central=2.505, fixed=True, label='60Co total gamma energy, MeV')
    energy_scale_normalization_point.transformations.front().setLabel('Energy scale normalization point')

    ns.printparameters(labels=True)

    #
    # Birk's model integration
    #
    binwidth=0.025
    epos_edges_full_input = N.arange(0.0, 12.0+1.e-6, binwidth)
    epos_firstbin = N.where(epos_edges_full_input>doubleme)[0][0]-1
    epos_edges_input=epos_edges_full_input[epos_firstbin:]
    epos_edges_input[0]=doubleme

    integrator_epos = C.IntegratorGL(epos_edges_input, 3, labels=('Evis sampler (GL)', 'Evis integrator (GL)'))
    epos_edges = integrator_epos.points.xedges
    epos_edges_full = C.Points(epos_edges_full_input)

    birks_e_input, birks_quenching_input = args.stoppingpower['e'], args.stoppingpower['dedx']
    birks_e_p, birks_quenching_p = C.Points(birks_e_input, labels='Te (input)'), C.Points(birks_quenching_input, labels='Stopping power (dE/dx)')

    with nsb:
        birks_integrand_raw = C.PolyRatio([], ['Kb0', 'Kb1', 'Kb2'], labels="Birk's integrand")
    birks_quenching_p >> birks_integrand_raw.polyratio.points

    doubleemass_point = C.Points([-doubleme], labels='2me offset')

    ekin_edges_p = C.SumBroadcast([epos_edges, doubleemass_point.points.points], labels='Evis to Te')
    ekin_edges_h = C.PointsToHist(ekin_edges_p, labels='Te bin edges')

    integrator_ekin = C.IntegratorGL(ekin_edges_h.adapter.hist, 2, labels=(('Te sampler (GL)', "Birk's integrator (GL)")))

    birks_integrand_interpolator = C.InterpLinear(birks_e_p, integrator_ekin.points.x, labels=("Birk's InSegment", "Birk's interpolator"))
    birks_integrand_interpolated = birks_integrand_interpolator.add_input(birks_integrand_raw.polyratio.ratio)
    birks_integral = integrator_ekin.add_input(birks_integrand_interpolated)

    birks_accumulator = C.PartialSum(0., labels="Birk's Evis\n[MeV]")
    birks_integral >> birks_accumulator.reduction

    #
    # Cherenkov model
    #
    with nsc:
        cherenkov = C.Cherenkov_Borexino(labels='Npe Cherenkov')
    ekin_edges_p >> cherenkov.cherenkov

    #
    # Electron energy model
    #
    with ns:
        electron_model = C.WeightedSum(['kC', 'Npesc'], [cherenkov.cherenkov.ch_npe, birks_accumulator.reduction.out], labels='Npe: electron responce')

    #
    # 2 511 keV gamma model
    #
    egamma_edges, egamma_x, egamma_birks_e_p, egamma_h, egamma_hp = load_electron_distribution(*args.annihilation_electrons)
    egamma_offset = 0
    lastpoint = N.where(ekin_edges_p.data()>egamma_edges[-1])[0][0]+1
    ekin_edges_lowe = C.View(ekin_edges_p, egamma_offset, lastpoint-egamma_offset, labels='Te Low E view')
    electron_model_lowe = C.View(electron_model.single(), egamma_offset, lastpoint-egamma_offset, labels='Npe: electron responce\n(low E view)')

    electron_model_lowe_interpolator = C.InterpLinear(ekin_edges_lowe.view.view, egamma_birks_e_p, labels=('Annihilation E InSegment', 'Annihilation gamma interpolator'))
    electron_model_lowe_interpolated = electron_model_lowe_interpolator.add_input(electron_model_lowe.view.view)

    with ns:
        npe_positron_offset = C.NormalizedConvolution('ngamma', labels='e+e- annihilation Evis [MeV]')
        electron_model_lowe_interpolated >> npe_positron_offset.normconvolution.fcn
        egamma_hp >> npe_positron_offset.normconvolution.weights

    #
    # Total positron model
    #
    positron_model = C.SumBroadcast([electron_model.sum.sum, npe_positron_offset.normconvolution.result],
                                    labels='Npe: positron responce')

    positron_model_scaled = C.FixedPointScale(epos_edges, energy_scale_normalization_point, labels=('Fixed point index', 'Positron energy model\nEvis, MeV'))
    positron_model_scaled = positron_model_scaled.add_input(positron_model.sum.outputs[0])

    #
    # Relative positron model
    #
    positron_model_relative = C.Ratio(positron_model_scaled, epos_edges, labels='Positron energy nonlinearity')
    positron_model_relative_full = C.ViewRear(epos_edges_full, epos_firstbin, epos_edges_input.size, 0.0, labels='Positron Energy nonlinearity')
    positron_model_relative >> positron_model_relative_full.view.rear

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

    birks_quenching_p.points.points.plot_vs(birks_e_p.points.points, '-', markerfacecolor='none', markersize=2.0, label='input')
    ax.legend(loc='upper right')
    savefig(args.output, suffix='_spower')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( "Birk's integrand" )

    birks_integrand_raw.polyratio.ratio.plot_vs(birks_e_p.points.points, '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='raw')
    # birks_integrand_interpolated.plot_vs(integrator_ekin.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    lines=birks_integrand_interpolated.plot_vs(integrator_ekin.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    birks_integrand_interpolated.plot_vs(integrator_ekin.points.x, 'o', alpha=0.5, color='black', markersize=0.6)
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

    birks_integral.plot_hist()

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
    birks_accumulator.reduction.plot_vs(ekin_edges_p)
    ax.plot([0.0, 12.0], [0.0, 12.0], '--', alpha=0.5)

    savefig(args.output, suffix='_birks_evis')

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)
    savefig()

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Cherenkov photons' )
    cherenkov.cherenkov.ch_npe.plot_vs(ekin_edges_p)

    savefig(args.output, suffix='_cherenkov_npe')

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 200.0)
    savefig()

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Electron model' )
    electron_model.single().plot_vs(ekin_edges_p)

    savefig(args.output, suffix='_electron')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Electron model (low energy view)' )
    annihilation_gamma_evis = npe_positron_offset.normconvolution.result.data()[0]
    label = 'Annihilation contribution=%.2f Npe'%annihilation_gamma_evis
    electron_model_lowe.single().plot_vs(ekin_edges_lowe.view.view, 'o', markerfacecolor='none', label='data')
    electron_model_lowe_interpolated.single().plot_vs(egamma_birks_e_p.single(), '-', label='interpolation\n'+label)

    ax.legend(loc='upper left')

    savefig(args.output, suffix='_electron_lowe')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Npe' )
    ax.set_title( 'Total Npe' )
    positron_model.sum.outputs[0].plot_vs(epos_edges)

    savefig(args.output, suffix='_total_npe')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Positron energy model' )
    positron_model_scaled.plot_vs(epos_edges)

    savefig(args.output, suffix='_total')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis/Edep' )
    ax.set_title( 'Positron energy nonlineairty' )
    positron_model_relative.single().plot_vs(epos_edges, label='nonlinearity')
    positron_model_relative_full.single().plot_vs(epos_edges_full.single(), '--', label='nonlinearity (full range)')

    savefig(args.output, suffix='_total_relative')

    ax.set_xlim(left=1.0)
    ax.set_ylim(0.5, 1.7)
    savefig(args.output, suffix='_total_relative1')

    if pdfpages:
        pdfpages.__exit__(None,None,None)

    savegraph(epos_edges, args.graph, namespace=ns)

    if args.show:
        P.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('stoppingpower', type=lambda fname: N.loadtxt(fname, dtype=dtype_spower))
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('-s', '--show', action='store_true', help='Show the plots')
    parser.add_argument('--annihilation-electrons', nargs=2, default=('input/hgamma2e.root', 'hgamma2e_1KeV'), help='Input electron distribution from 2 511 keV gammas')

    main( parser.parse_args() )
