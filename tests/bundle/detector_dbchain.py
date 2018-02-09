#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundle
import numpy as N
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import constructors as C
from converters import convert
from gna.configurator import NestedDict, uncertaindict, uncertain
import itertools as I
from physlib import percent
from gna import parameters

def unit_bins_at( values, edges ):
    """For given x-values and bin edges returns a histogram with unit bins containing x-values"""
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

#
# Parse arguments
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
args = parser.parse_args()

#
# Define the configuration
#
cfg = NestedDict()
cfg.detector = NestedDict(
        bundle = 'bundlechain_v01',
        detectors = [ 'AD11', 'AD21', 'AD31' ],
        chain = [ 'iav', 'nonlinearity', 'eres', 'rebin' ],
        )
cfg.detector.nonlinearity = NestedDict(
        bundle = 'detector_nonlinearity_db_root_v01',
        names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
        filename = 'data/dayabay/tmp/detector_nl_consModel_450itr.root',
        parname = 'escale.{}',
        par = uncertain(1.0, 0.2, 'percent'),
        )
cfg.detector.iav = NestedDict(
        bundle = 'detector_iav_db_root_v01',
        parname = 'OffdiagScale.{}',
        scale   = uncertain(1.0, 4, 'percent'),
        ndiag = 1,
        filename = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
        matrixname = 'iav_matrix'
        )
cfg.detector.eres = NestedDict(
        bundle = 'detector_eres_common3',
        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
        pars = uncertaindict(
            [('Eres_a', 0.014764) ,
             ('Eres_b', 0.0869) ,
             ('Eres_c', 0.0271)],
            mode='percent',
            uncertainty=30
            )
        )
cfg.detector.rebin = NestedDict(
        bundle = 'rebin',
        rounding = 3,
        edges = [ 0.0, 5.0, 10.0 ]
        )

#
# Define namespaces
#
namespaces = cfg.detector.detectors

#
# Bin edges, required by energy nonlinearity
# Data input
#
nbins = 240
edges = N.linspace(0.0, 12.0, nbins+1, dtype='d')
points = C.Points(edges)

hists_list = ()
for eset in ( (1.025, 6.025), (2.025, 7.025), (3.025, 8.025) ):
    heights = unit_bins_at( eset, edges )
    hist = C.Histogram( edges, heights )
    hists_list += hist,

#
# Define the chain
#
b, = execute_bundle(edges=points.single(), cfg=cfg.detector, namespaces=namespaces)

from gna.parameters.printer import print_parameters
print_parameters( env.globalns )

#
# Connect inputs
#
for inp, hist in zip(b.inputs.values(), hists_list):
    inp( hist.hist )

#
# Dump graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot( b.transformations_out[0][0] )
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )

#
# Make plots
#
from mpl_tools.helpers import plot_hist
from gna.labelfmt import formatter as L
axes=()
for i, hist in enumerate(hists_list):
    P.figure(),
    ax = P.subplot( 111 )
    axes+=ax,
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( L.u('evis') )
    ax.set_ylabel( 'Entries' )
    ax.set_title( cfg.detector.detectors[i] )

    data = hist.hist.hist.data()
    print( 'Sum data %i=%i'%( i, data.sum() ) )
    plot_hist(edges, data, label='original')

for bundle in b.bundles.values():
    for i, (oname, out) in enumerate( bundle.outputs.items() ):
        P.sca(axes[i])

        data = out.data()
        print( 'Sum data %s (%s) %i=%f'%( type(bundle).__name__, oname, i, data.sum() ) )

        plot_hist(out.datatype().edges, data, label=type(bundle).__name__)

for i, hist in enumerate(hists_list):
    ax=axes[i]
    P.sca(ax)
    ax.legend( loc='upper right' )

P.show()
