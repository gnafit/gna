#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C

parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
parser.add_argument( '-s', '--show', action='store_true' )
opts = parser.parse_args()

env.defparameter("E_0", central=0.165, fixed=True)
env.defparameter("p0",  central=-7.26624e+00, fixed=True)
env.defparameter("p1",  central=1.72463e+01,  fixed=True)
env.defparameter("p2",  central=-2.18044e+01, fixed=True)
env.defparameter("p3",  central=1.44731e+01,  fixed=True)
env.defparameter("p4",  central=3.22121e-02,  fixed=True)

energy = N.linspace(0.166, 10, num=1000)
gna_energy = C.Points(energy)
cherenkov = R.Cherenkov()
cherenkov.cherenkov.energy(gna_energy)
ch_response = cherenkov.cherenkov.ch_npe.data()

fig = P.figure()
ax = P.subplot( 111 )
ax.plot(energy, ch_response)
ax.set_yscale("log")
ax.set_xscale("log")
if opts.show:
    P.show()

#  fig = P.figure()
#  ax = P.subplot( 111 )
#  ax.minorticks_on()
#  ax.grid()
#  ax.set_xlabel( L.u( 'edep' ) )
#  ax.set_ylabel( 'Correction' )
#  ax.set_title( 'Non-linearity correction' )

#  ax.plot( edges, corr )

#  savefig( opts.output, suffix='_corr' )

#  fig = P.figure()
#  ax = P.subplot( 111 )
#  ax.minorticks_on()
#  ax.grid()
#  ax.set_xlabel( L.u( 'edep' ) )
#  ax.set_ylabel( L.u( 'evis' ) )
#  ax.set_title( 'Non-linearity correction' )

#  ax.plot( edges, edges_m_plot )
#  ax.plot( [ edges[0], edges[-1] ], [ edges[0], edges[-1] ], '--' )

#  savefig( opts.output, suffix='_energy' )

#  pedges_m = C.Points( edges_m )
#  ev = [ 1.025, 2.025, 3.025, 5.025, 6.025, 9.025 ]
#  phist = singularities( ev, edges )
#  hist = C.Histogram( edges, phist )

#  nl = R.HistNonlinearity()
#  nl.print()

#  nl.set(hist.hist, pedges_m.points)

#  nl.add_input(hist)

#  smeared = nl.smear.Nrec.data()
#  print( 'Sum check (diff): {}'.format( phist.sum()-smeared.sum() ) )

#  fig = P.figure()
#  ax = P.subplot( 111 )
#  ax.minorticks_on()
#  ax.grid()
#  ax.set_xlabel( L.u('evis') )
#  ax.set_ylabel( 'Entries' )
#  ax.set_title( 'Non-linearity correction' )

#  lines = plot_hist( edges, smeared )
#  color = lines[0].get_color()
#  _, ev_m = nlfcn(ev)
#  heights = smeared[smeared>0.45]
#  ax.vlines( ev,   0.0, heights, alpha=0.7, color='red', linestyle='--' )
#  ax.vlines( ev_m, 0.0, heights, alpha=0.7, color='green', linestyle='--' )

#  savefig( opts.output, suffix='_evis' )

#  fig = P.figure()
#  ax1 = P.subplot( 111 )
#  ax1.minorticks_on()
#  ax1.grid()
#  ax1.set_xlabel( 'Source bins' )
#  ax1.set_ylabel( 'Target bins' )
#  ax1.set_title( 'Energy non-linearity matrix' )

#  mat = convert(nl.getDenseMatrix(), 'matrix')
#  print( 'Col sum', mat.sum(axis=0) )

#  mat = N.ma.array( mat, mask= mat==0.0 )
#  c = ax1.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
#  add_colorbar( c )

#  ax1.plot( edges, edges_m_plot, '--', linewidth=0.5, color='white' )
#  ax1.set_ylim( edges[-1], edges[0] )

#  savefig( opts.output, suffix='_mat' )

#  if opts.show:
    #  P.show()
