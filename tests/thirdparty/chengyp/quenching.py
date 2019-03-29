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
    buf = hist.get_buffer_hist1(hist)
    edges = hist.get_bin_edges_axis(hist.GetXaxis())

def main(args):
    ns = env.globalns('energy')
    nsb = ns('birks')
    nsb.defparameter('Kb0', central=1.0, fixed=True, label='Kb0=1')
    nsb.defparameter('Kb1', central=0.0062, fixed=True, label="Birk's constant")
    nsb.defparameter('Kb2', central=1.5e-6, fixed=True, label="Birk's constant (second derivative)")
    nsc = ns('cherenkov')
    nsc.defparameter("E_0", central=0.165, fixed=True)
    nsc.defparameter("p0",  central=-7.26624e+00, fixed=True)
    nsc.defparameter("p1",  central=1.72463e+01,  fixed=True)
    nsc.defparameter("p2",  central=-2.18044e+01, fixed=True)
    nsc.defparameter("p3",  central=1.44731e+01,  fixed=True)
    nsc.defparameter("p4",  central=3.22121e-02,  fixed=True)

    Nsc = ns.defparameter("Nsc", central=1341.38, fixed=True)
    Nch = ns.defparameter("Nch", central=1., fixed=True)
    Eannihil = ns.defparameter("Eannihil", central=2.0*0.511, fixed=True, label='Electron-positron annihilation energy')

    ns.printparameters(labels=True)

    binwidth=0.025
    bins = N.arange(0.0, 12.0+1.e-6, binwidth)
    xa, dedx = args.stoppingpower['e'], args.stoppingpower['dedx']
    xp, dedx_p = C.Points(xa, labels='Energy'), C.Points(dedx, labels='Stopping power (dE/dx)')

    earg_view = C.View(xp, 0, int(0.600001//binwidth), labels='Low E view')

    with nsb:
        pratio = C.PolyRatio([], ['Kb0', 'Kb1', 'Kb2'], labels="Birk's integrand")
    dedx_p >> pratio.polyratio.points

    integrator = C.IntegratorGL(bins, 4, labels=('GL sampler', 'GL integrator'))
    interpolator = C.InterpLinear(xp, integrator.points.x, labels=('InSegment', 'Interpolator'))
    interpolated = interpolator.add_input(pratio.polyratio.ratio)
    integrated = integrator.add_input(interpolated)

    accumulator = C.PartialSum(labels="Birk's contribution\n(absolute)")
    accumulator.reduction << integrated

    with nsc:
        cherenkov = C.Cherenkov(labels='Cherenkov contribution\n(absolute)')
    cherenkov.cherenkov << integrator.points.xcenters

    with ns:
        npe_electron = C.WeightedSum(['Nch', 'Nsc'], [cherenkov.cherenkov.ch_npe, accumulator.reduction.out], labels='{Electron energy model|(absolute)}')

    savegraph(xp, args.graph, namespace=ns)

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'dE/dx' )
    ax.set_title( 'Stopping power' )

    dedx_p.points.points.plot_vs(xp.points.points, '-', markerfacecolor='none', markersize=2.0, label='input')
    ax.legend(loc='upper right')
    savefig(args.output, suffix='_spower')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Integrand' )

    pratio.polyratio.ratio.plot_vs(xp.points.points, '-', markerfacecolor='none', markersize=2.0, label='raw')
    interpolated.plot_vs(integrator.points.x, '-', markerfacecolor='none', markersize=2.0, label='interpolated')
    ax.legend(loc='upper right')
    savefig(args.output, suffix='_spower')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( '' )
    ax.set_title( 'Integrated' )

    integrated.plot_hist()
    savefig(args.output, suffix='_spower_int')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Partial sum' )
    ax.set_title( 'Partial sum integrated' )
    #  accumulator.reduction.out.plot_vs(integrator.transformations.hist, '-', markerfacecolor='none', markersize=2.0, label='partial sum')
    accumulator.reduction.plot_hist()

    P.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('stoppingpower', type=lambda fname: N.loadtxt(fname, dtype=dtype_spower))
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('--annihilation-electrons', nargs=2, default=('input/hgamma2e.root', 'hgamma2e_1KeV'), help='Input electron distribution from 2 511 keV gammas')

    main( parser.parse_args() )
