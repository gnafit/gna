#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundle
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist
from gna.env import env
import constructors as C
import numpy as N
from gna.configurator import NestedDict, uncertain
from physlib import percent

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
args = parser.parse_args()

#
# Initialize bundle
#
cfg = NestedDict(
        bundle = 'detector_iav_db_root_v01',
        parname = 'OffdiagScale.{}',
        scale   = uncertain(1.0, 4, 'percent'),
        ndiag = 1,
        filename = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
        matrixname = 'iav_matrix'
        )
b, = execute_bundle( cfg=cfg, namespaces=['AD1', 'AD2', 'AD3'] )
(smear1, smear2, smear3) = b.transformations_out.values()

env.globalns.printparameters(labels=True)

par1, par2, par3 = (b.common_namespace('OffdiagScale')[s] for s in ('AD1', 'AD2', 'AD3'))
par1.set( 1.5 )
par2.set( 1.0 )
par3.set( 0.5 )

#
# Test bundle
#
def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

binwidth=0.05
edges = N.arange( 0.0, 12.0001, binwidth )

phist = singularities( [ 1.225, 4.025, 7.025 ], edges )
hist = C.Histogram( edges, phist )
smear1.inputs.Ntrue( hist.hist )
smear2.inputs.Ntrue( hist.hist )
smear3.inputs.Ntrue( hist.hist )

#
# Plot
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'IAV effect' )

smeared1 = smear1.Nvis.data()
smeared2 = smear2.Nvis.data()
smeared3 = smear3.Nvis.data()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared1.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared2.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared3.sum() ) )

lines = plot_hist( edges, smeared1, linewidth=1.0, label='IAV 1' )
lines = plot_hist( edges, smeared2, linewidth=1.0, label='IAV 2' )
lines = plot_hist( edges, smeared3, linewidth=1.0, label='IAV 3' )

ax.legend( loc='upper right' )

#
# Dump graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot( b.transformations_out.values()[0] )
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )

if args.show:
    P.show()
