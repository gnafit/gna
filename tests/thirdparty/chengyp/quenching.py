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

dtype_spower = [ ('e', 'd'), ('temp1', 'd'), ('temp2', 'd'), ('dedx', 'd') ]

def main(args):
    bins = N.arange(0.0, 12.0+1.e-6, 0.025)
    xa, ya = args.stoppingpower['e'], args.stoppingpower['dedx']
    xp, yp = C.Points(xa, labels='Energy'), C.Points(ya, labels='Stopping power')

    integrator = C.IntegratorGL(bins, 2, labels=('GL sampler', 'GL integrator'))
    interpolator = C.InterpLinear(xp, integrator.points.x, labels=('InSegment', 'Interpolator'))
    interpolated = interpolator.add_input(yp)
    integrated = integrator.add_input(interpolated)

    savegraph(xp, args.graph)

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Stopping power' )
    ax.set_title( '' )

    yp.points.points.plot_vs(xp.points.points, '-', markerfacecolor='none', markersize=2.0, label='input')
    interpolated.plot_vs(integrator.points.x, 'o', markerfacecolor='none', markersize=2.0, label='sample points')
    ax.legend(loc='upper right')
    savefig(args.output, suffix='_spower')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'E, MeV' )
    ax.set_ylabel( 'Stopping power' )
    ax.set_title( 'Integrated' )

    integrated.plot_hist()
    savefig(args.output, suffix='_spower_int')

    P.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('stoppingpower', type=lambda fname: N.loadtxt(fname, dtype=dtype_spower))
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-g', '--graph', help='Output file for graph')

    main( parser.parse_args() )
