#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for the detector_nl bundle with two detectors"""

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle.detector_nl import detector_nl_from_file
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import constructors as C
from converters import convert
import numpy as N

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument( '-o', '--output' )
# opts=parser.parse_args()

#
# Initialize bundle
#
names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3'  ]
pars = [ env.defparameter( 'weight_'+names[0], central=1.0, sigma=0.0, fixed=True ) ]
for name in names[1:]:
    par = env.defparameter( 'weight_'+name, central=0.0, sigma=1.0 )
    pars.append( par )

escale1 = env.defparameter( 'ad1.escale', central=1.0, sigma=0.02*0.01 )
escale2 = env.defparameter( 'ad2.escale', central=1.0, sigma=0.02*0.01 )

escale1.set(0.98)
escale2.set(1.02)

def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

nbins = 240
edges = N.linspace(0.0, 12.0, nbins+1, dtype='d')
edges_p = C.Points(edges)
phist = singularities( [ 1.225, 2.225, 4.025, 7.025, 9.025 ], edges )
hist = C.Histogram( edges, phist )

filename = 'output/detector_nl_consModel_450itr.root'
(nonlin1, nonlin2), storage = detector_nl_from_file( filename, names, edges=edges_p.points,
                                         namespaces=[ env.ns(ns) for ns in ('ad1', 'ad2') ],
                                         debug=True )
factor1 = storage('escale_ad1')['factor']
factor2 = storage('escale_ad2')['factor']

#
# Plot curves:
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( '' )

lines = ax.plot( edges, factor1.data(), '-', label='Escale 1' )
lines = ax.plot( edges, factor2.data(), '-', label='Escale 2' )

ax.legend( loc='lower right' )

#
# Test bundle
#
nonlin1.smear.Ntrue( hist.hist )
nonlin2.smear.Ntrue( hist.hist )

#
# Plot hists
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'Non-linearity effect' )

smeared1 = nonlin1.smear.Nvis.data().copy()
smeared2 = nonlin2.smear.Nvis.data().copy()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared1.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared2.sum() ) )

lines = plot_hist( edges, smeared1, label='Smeared 1' )
lines = plot_hist( edges, smeared2, label='Smeared 2' )
plot_hist( edges, phist, label='original' )

ax.legend( loc='upper right' )

P.show()
