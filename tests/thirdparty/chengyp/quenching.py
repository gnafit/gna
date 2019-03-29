#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna.bindings import common
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.graphviz import savegraph
from gna.env import env

dtype_spower = [ ('e', 'd'), ('temp1', 'd'), ('temp2', 'd'), ('dedx', 'd') ]

def load_electron_distribution(filename, objname):
    file = R.TFile(filename, 'read')
    hist = file.Get(objname)
    buf = get_buffer_hist1(hist).copy()
    buf/=buf.sum()
    edges = get_bin_edges_axis(hist.GetXaxis())
    centers = (edges[:-1] + edges[1:])*0.5

    return edges, centers, C.Points(centers, labels='gamma-e centers'), buf, C.Points(buf, labels='gamma-e weights')

def main(args):
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

    Nsc = ns.defparameter("Nphsc", central=1341.38, fixed=True, label='Scintillation responce, Nph/MeV')
    Nch = ns.defparameter("kC", central=1., fixed=True, label='Cerenkov contribution normalization')

    from physlib import pdg
    emass = ns.defparameter("emass", central=pdg['live']['ElectronMass'], fixed=True, label='Electron mass, MeV')
    ns.defparameter("ngamma", central=2.0, fixed=True, label='Number of e+e- annihilation gammas')

    ns.printparameters(labels=True)

    #
    # Birk's model integration
    #
    binwidth=0.025
    bins = N.arange(0.0, 12.0+1.e-6, binwidth)

    xa, dedx = args.stoppingpower['e'], args.stoppingpower['dedx']
    xp, dedx_p = C.Points(xa, labels='Energy'), C.Points(dedx, labels='Stopping power (dE/dx)')

    with nsb:
        pratio = C.PolyRatio([], ['Kb0', 'Kb1', 'Kb2'], labels="Birk's integrand")
    dedx_p >> pratio.polyratio.points

    integrator = C.IntegratorGL(bins, 3, labels=('Evis sampler (GL)', 'Evis integrator (GL)'))

    emass_point = C.Points([-2*emass.value()], labels='2me offset')
    inputs = [emass_point.points.points, integrator.points.x]

    ekin_points = C.SumBroadcast(inputs, labels='Evis to Te')
    ekin_edges = C.PointsToHist(ekin_points, -2., labels='Te bin edges')

    ekin_integrator = R.IntegratorGL(len(ekin_edges.adapter.hist.data()), 2, labels=(('Te sampler (GL)', 'Te integrator (GL)')))
    ekin_integrator.points.edges(ekin_edges.adapter.hist)

    interpolator = C.InterpLinear(xp, ekin_integrator.points.x, labels=('InSegment', 'Interpolator'))
    interpolated = interpolator.add_input(pratio.polyratio.ratio)
    integrated = ekin_integrator.add_input(interpolated)

    accumulator = C.PartialSum(0., labels="Evis (Birks)\n[MeV]")
    accumulator.reduction << integrated

    #
    # Cherenkov model
    #
    electron_model_e = ekin_integrator.points.xcenters

    with nsc:
        cherenkov = C.Cherenkov_Borexino(labels='Nph Cherenkov')
    cherenkov.cherenkov << electron_model_e

    #
    # Electron energy model
    #
    with ns:
        electron_model = C.WeightedSum(['kC', 'Nphsc'], [cherenkov.cherenkov.ch_npe, accumulator.reduction.out], labels='Nph: electron responce')

    #
    # 2 511 keV gamma model
    #
    egamma_edges, egamma_x, egamma_xp, egamma_h, egamma_hp = load_electron_distribution(*args.annihilation_electrons)
    elow_npoints = int(egamma_edges[-1]*0.99999999//binwidth)
    electron_model_elow = C.View(electron_model_e, 0, elow_npoints, labels='Low E view')
    electron_model_lowe = C.View(electron_model, 0, elow_npoints, labels='Nph: electron responce\n(low E view)')

    interpolator_2g = C.InterpLinear(electron_model_elow.view.view, egamma_xp, labels=('InSegment', 'e-gamma interpolator'))
    electron_model_lowe_interpolated = interpolator_2g.add_input(electron_model_lowe.view.view)

    with ns:
        npe_positron_offset = C.NormalizedConvolution('ngamma', labels='e+e- annihilation E')
        electron_model_lowe_interpolated >> npe_positron_offset.normconvolution.fcn
        egamma_hp >> npe_positron_offset.normconvolution.weights

    #
    # Total positron model
    #
    positron_model = C.SumBroadcast([electron_model.sum.sum, npe_positron_offset.normconvolution.result],
                                    labels='Nph: positron responce')

    # fixed_point = C.Points([2.0*emass.value()], labels='{Energy scale normalization|(2me)}')
    fixed_point = C.Points([4.0], labels='{Energy scale normalization|(4 MeV)}')
    positron_model_relative = C.FixedPointScale(integrator.points.x, fixed_point.points.points, labels=('Fixed point index', 'Positron energy nonlinearity'))
    positron_model_relative_out = positron_model_relative.add_input(positron_model.sum.outputs[0])

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

    dedx_p.points.points.plot_vs(xp.points.points, '-', markerfacecolor='none', markersize=2.0, label='input')
    ax.legend(loc='upper right')
    savefig(args.output, suffix='_spower')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Integrand' )

    pratio.polyratio.ratio.plot_vs(xp.points.points, '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='raw')
    interpolated.plot_vs(ekin_integrator.points.x,   '-', alpha=0.5, markerfacecolor='none', markersize=2.0, label='interpolated')
    ax.legend(loc='lower right')
    savefig(args.output, suffix='_spower_int')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Integrated' )

    integrated.plot_hist()
    savefig(args.output, suffix='_spower_intc')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Edep, MeV' )
    ax.set_ylabel( 'Evis, MeV' )
    ax.set_title( 'Electron energy (Birks)' )
    #  accumulator.reduction.out.plot_vs(integrator.transformations.hist, '-', markerfacecolor='none', markersize=2.0, label='partial sum')
    accumulator.reduction.plot_hist()

    savefig(args.output, suffix='_birks_evis')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Nph' )
    ax.set_title( 'Cherenkov photons' )
    cherenkov.cherenkov.ch_npe.plot_vs(electron_model_e)

    savefig(args.output, suffix='_cherenkov_nph')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Nph' )
    ax.set_title( 'Total Nph' )
    positron_model.sum.outputs[0].plot_vs(integrator.points.x)

    savefig(args.output, suffix='_total_nph')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Evis/Etrue' )
    ax.set_title( 'Positron energy nonlineairty' )
    positron_model_relative_out.plot_vs(integrator.points.x)

    savefig(args.output, suffix='_total')

    P.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('stoppingpower', type=lambda fname: N.loadtxt(fname, dtype=dtype_spower))
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('--annihilation-electrons', nargs=2, default=('input/hgamma2e.root', 'hgamma2e_1KeV'), help='Input electron distribution from 2 511 keV gammas')

    main( parser.parse_args() )