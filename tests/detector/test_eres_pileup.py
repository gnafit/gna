#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools import bindings
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C
from gna.parameters.printer import print_parameters
from gna.bindings import common

parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
parser.add_argument( '-s', '--show', action='store_true' )
opts = parser.parse_args()

def axes( title, ylabel='' ):
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( L.u('evis') )
    ax.set_ylabel( ylabel )
    ax.set_title( title )
    return ax

#
# Define the parameters in the current namespace
#
percent = 0.01
env.defparameter( 'Eres_a',  central=0.0, fixed=True )
par = env.defparameter( 'Eres_b',  central=0.03, fixed=True )
env.defparameter( 'Eres_c',  central=0.0, fixed=True )
print_parameters( env.globalns )

#
# Define bin edges
#
nbins=40
edges = N.linspace(0.0, 2.0, nbins+1)
binwidth=edges[1]-edges[0]
centers = (edges[1:]+edges[:-1])*0.5
efine = N.arange( edges[0], edges[-1]+1.e-5, 0.005 )

ax = axes( 'Energy resolution impact' )
phist = N.ones(edges.size-1)

hist = C.Histogram(edges, phist)
edges_o = R.HistEdges(hist)
eres = R.EnergyResolution(True)
eres.matrix.Edges( hist )
eres.smear.Ntrue( hist )
smeared = eres.smear.Nrec.data()

eres.smear.plot_hist(label='default')

arrays = N.eye(20, edges.size-1, dtype='d')
objects = [C.Histogram(edges, a) for a in arrays]
outputs = [o.single() for o in objects]

for i, out in enumerate(outputs):
    out = eres.add_input(out)

    out.plot_hist(label='Bin %i'%i, alpha=0.5)

ax.legend()

savefig(opts.output)

smeared = eres.smear.Nrec.data()

ax = axes( 'Relative energy uncertainty', ylabel=L.u('eres_sigma_rel') )
x = N.arange( 0.5, 12.0, 0.01 )
fcn = N.frompyfunc( eres.relativeSigma, 1, 1 )
y = fcn( x )

ax.plot( x, y*100. )
savefig( opts.output, suffix='_sigma' )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'Energy resolution convertsion matrix (class)' )

mat = convert(eres.getDenseMatrix(), 'matrix')
mat = N.ma.array( mat, mask= mat==0.0 )
c = ax.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
add_colorbar( c )

savefig( opts.output, suffix='_matc' )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'Energy resolution convertsion matrix (trans)' )

eres.matrix.FakeMatrix.plot_matshow(colorbar=True, mask=0.0, extent=[edges[0], edges[-1], edges[-1], edges[0]])

savefig( opts.output, suffix='_mat' )

if opts.show:
    P.show()

