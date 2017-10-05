#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from converters import convert
from argparse import ArgumentParser

edges   = N.array( [  0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ] )
edges_m = N.array( [ -1.0, 0.5, 1.2, 1.8, 4.0, 5.0, 6.2, 7.5 ] )

pedges, pedges_m = convert( edges, 'points' ), convert( edges_m, 'points' )
ntrue = R.Histogram( edges.size-1, edges, N.ones( edges.size-1 ) )

nl = R.EnergyNonlinearity()
nl.set( pedges, pedges_m, ntrue )

idy = R.Identity()
idy.identity.source(nl.matrix.Matrix)

mat = idy.identity.target.data()
print( mat )
print( mat.sum( axis=0 ) )

import IPython
IPython.embed()
