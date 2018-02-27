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
from gna.labelfmt import formatter as L

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-l', '--log', action='store_true', help='logarithmic scale' )
opts=parser.parse_args()

cfg = NestedDict()
cfg.bundle = 'reactor_anu_spectra_v01'
cfg.isotopes = [ 'U5', 'U8', 'Pu9', 'Pu1' ]
cfg.filename = ['data/Huber_smooth_extrap_{iso}_13MeV0.01MeVbin.dat', 'data/Mueller_smooth_extrap_{iso}_13MeV0.01MeVbin.dat']
cfg.strategy = dict( underflow='constant', overflow='extrapolate' )
cfg.edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) )

points = N.linspace( 0.0, 12.0, 241 )
points_t = C.Points(points)

shared=NestedDict( points=points_t.single() )
b, = execute_bundle( cfg=cfg, shared=shared )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( L.u('anu_yield') )
ax.set_title( '' )

ax.vlines(cfg.edges, 0.0, 2.5, linestyles='--', alpha=0.5, colors='blue')

for name, output in b.outputs.items():
    ax.plot( points, output.data(), label=L.s(name) )

if opts.log:
    ax.set_yscale('log')

ax.legend( loc='upper right' )

P.show()
