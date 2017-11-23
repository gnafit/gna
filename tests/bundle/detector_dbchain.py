#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundle
from gna.bundle import all
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import constructors as C
from converters import convert
import numpy as N
from gna.configurator import NestedDict
import itertools as I
from physlib import percent

#
# Parse arguments
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
args = parser.parse_args()

#
# Create the configuration
#
cfg = NestedDict()
cfg.detector = dict(
        bundle = 'dbchain_v01',
        detectors = [ 'AD11', 'AD21', 'AD31' ],
        chain = [ 'iav', 'nonlinearity', 'resolution', 'rebin' ]
        )
cfg.detector.nonlinearity = dict(
        bundle = 'nonlinearity_db_root_v01',
        names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
        filename = 'output/detector_nl_consModel_450itr.root',
        relative_uncertainty = 0.2*percent
        )
cfg.detector.iav = dict(
        bundle = 'iav_db_root_v01',
        parname = 'OffdiagScale',
        offdiag_scale_uncertainty = 4*percent,
        ndiag = 1,
        filename = 'output/detector_iavMatrix_P14A_LS.root',
        matrixname = 'iav_matrix'
        )

#
# Create namespaces
#
namespaces = cfg.detector.detectors
storage = env.globalns('storage')

#
# Create variables
#
env.defparameter( 'weight_'+cfg.detector.nonlinearity.names[0], central=1.0, sigma=0.0, fixed=True )
for name in cfg.detector.nonlinearity.names[1:]:
    env.defparameter( 'weight_'+name, central=0.0, sigma=1.0 )

for detector in cfg.detector.detectors:
    env.defparameter( '{detector}.OffdiagScale'.format(detector=detector, parname=cfg.detector.iav.parname),
                      central=1.0, relsigma=cfg.detector.iav.offdiag_scale_uncertainty )
    env.defparameter( '{detector}.escale'.format(detector=detector),
                      central=1.0, relsigma=cfg.detector.nonlinearity.relative_uncertainty )


#
# Bin edges, required by energy nonlinearity
#
nbins = 240
edges = N.linspace(0.0, 12.0, nbins+1, dtype='d')
points = C.Points(edges)

#
# Create the chain
#
t, b = execute_bundle( edges=points.single(), cfg=cfg.detector, namespaces=namespaces, storage=storage  )

#
# Make common fake input
#
from constructors import Points
data = Points( N.ones( (240) ) )
for inp in b.inputs:
    inp.smear.Ntrue( data.points )

#
# Dump graph
#
if args.dot:
    from gna.graphviz import GNADot

    graph = GNADot( t[0]['smear'] )
    graph.write(args.dot)

