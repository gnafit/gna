#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna.bindings import common
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig
from gna.graphviz import savegraph
from gna.env import env

dtype_spower = [ ('e', 'd'), ('temp1', 'd'), ('temp2', 'd'), ('dedx', 'd') ]

def main(args):
    ns = env.globalns('birks')
    ns.defparameter('Kb0', central=1.0, fixed=True)
    ns.defparameter('Kb1', central=0.0062, fixed=True)
    ns.defparameter('Kb2', central=1.5e-6, fixed=True)
    ns.defparameter("E_0", central=0.165, fixed=True)
    ns.defparameter("p0",  central=-7.26624e+00, fixed=True)
    ns.defparameter("p1",  central=1.72463e+01,  fixed=True)
    ns.defparameter("p2",  central=-2.18044e+01, fixed=True)
    ns.defparameter("p3",  central=1.44731e+01,  fixed=True)
    ns.defparameter("p4",  central=3.22121e-02,  fixed=True)

    bins = N.arange(0.0, 12.0+1.e-6, 0.025)
    xa, dedx = args.stoppingpower['e'], args.stoppingpower['dedx']
    xp, dedx_p = C.Points(xa, labels='Energy'), C.Points(dedx, labels='Stopping power')

    with ns:
        pratio = C.PolyRatio([], ['Kb0', 'Kb1', 'Kb2'], labels='Integrand')
    dedx_p >> pratio.polyratio.points

    integrator = C.IntegratorGL(bins, 4, labels=('GL sampler', 'GL integrator'))
    interpolator = C.InterpLinear(xp, integrator.points.x, labels=('InSegment', 'Interpolator'))
    interpolated = interpolator.add_input(pratio.polyratio.ratio)
    integrated = integrator.add_input(interpolated)

    accumulator = C.PartialSum(labels='reduction')
    accumulator.reduction << integrated

    #  import IPython
    #  IPython.embed()

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

    main( parser.parse_args() )
