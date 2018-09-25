#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundle
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import constructors as C
import numpy as N
from gna.configurator import NestedDict, uncertain
from physlib import percent

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
#  parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-o', '--output', help='output file' )
#  parser.add_argument( '-x', '--xlim', nargs=2, type=float, help='xlim' )
args = parser.parse_args()

#DYB-like detector positions in meters
ADs = {
    'AD11' : [ 362.8329,    50.4206, -70.8174 ],
    'AD12' : [ 358.8044,    54.8583, -70.8135 ],
    'AD21' : [   7.6518,  -873.4882, -67.5241 ],
    'AD22' : [   9.5968,  -879.149,  -67.5202 ],
    'AD31' : [ 936.7486, -1419.013,  -66.4852 ],
    'AD32' : [ 941.4493, -1415.305,  -66.4966 ],
    'AD33' : [ 940.4612, -1423.737,  -66.4965 ],
    'AD34' : [ 945.1678, -1420.0282, -66.4851 ],
}

#DYB-like reactor positions in meters
reactors = {
    'DB1' : [  359.2029,  411.4896, -40.2308 ],
    'DB2' : [  448.0015,  411.0017, -40.2392 ],
    'LA1' : [ -319.666,  -540.7481, -39.7296 ],
    'LA2' : [ -267.0633, -469.2051, -39.7230 ],
    'LA3' : [ -543.284,  -954.7018, -39.7987 ],
    'LA4' : [ -490.6906, -883.152,  -39.7884 ],
}



#
# Initialize bundle
#
cfg = NestedDict(
    # Bundle name
    bundle = 'baselines',
    # Reactor positions
    reactors = reactors,
    # Detector positions
    detectors = ADs
    )
b, = execute_bundle(cfg=cfg)

env.globalns.printparameters(labels=True)


#
# Test bundle
#
