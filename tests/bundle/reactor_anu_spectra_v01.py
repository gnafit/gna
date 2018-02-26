#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundle
from gna.env import env, findname
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar
import constructors as C

cfg = NestedDict()
cfg.bundle = 'reactor_anu_spectra_v01'
cfg.isotopes = [ 'U5', 'U8', 'Pu9', 'Pu1' ]
cfg.filename = ['data/Huber_smooth_extrap_{iso}_13MeV0.01MeVbin.dat', 'data/Mueller_smooth_extrap_{iso}_13MeV0.01MeVbin.dat']
cfg.strategy = dict( underflow='fill', overflow='extrapolate' )
cfg.edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) )

points = N.linspace( 0.0, 12.0, 241 )
points_t = C.Points(points)

shared=NestedDict( points=points_t.single() )
b, = execute_bundle( cfg=cfg )
