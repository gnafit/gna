#!/usr/bin/env python

from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundles
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.env import env
import gna.constructors as C
import numpy as N
from gna.configurator import NestedDict, uncertain
from physlib import percent
from gna.expression import *

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--from-file', action="store_true", help='Read coordinates from files')
args = parser.parse_args()
args.make_idx=True


if args.from_file:
    reactors  = 'data/dayabay/reactor/coordinates/coordinates_docDB_9757.py'
    detectors = 'data/dayabay/ad/coordinates/coordinates_docDB_9757.py'
    snf_pools  = 'data/dayabay/snf/coordinates/coordinates_docDB_11112.py'
else:
    #DYB-like detector positions in meters
    detectors = {
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

    snf_pools = None


indices = [('r', 'reactor', ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']),
           ('d', 'detector', ['AD11', 'AD12', 'AD21', 'AD22', 'AD31',
                              'AD32', 'AD33', 'AD34'])]
if args.make_idx:
    expr = 'baselines[d,r]()'
    a =  Expression(expr, indices=indices)
    a.parse()
    lib = dict()
    a.guessname(lib, save=True)
    a.tree.dump(True)


cfg = NestedDict(
    # Bundle name
    bundle = dict(
        name = 'reactor_baselines',
        version = 'v01',
        nidx = indices,
        major = 'rd' # first for reactor, second for detector
        ),
    # Reactor positions
    reactors = reactors,
    # Detector positions
    detectors = detectors,
    # SNF pools,
    snf_pools=snf_pools,
    unit="meter"
    )
#
# Initialize bundle
#

if args.make_idx:
    cfg_idx = NestedDict(
            baselines=cfg)
    context = ExpressionContext(cfg_idx, ns=env.globalns)
    a.build(context)
else:
    b, = execute_bundles(cfg=cfg)

env.globalns.printparameters(labels=True)

#
# Test bundle
#
