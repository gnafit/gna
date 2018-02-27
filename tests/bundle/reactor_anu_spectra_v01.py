#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundle
from gna.env import env
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar
import constructors as C
from gna.labelfmt import formatter as L

"""Parse arguments"""
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-l', '--log', action='store_true', help='logarithmic scale' )
parser.add_argument( '--set', nargs=2, action='append', default=[], help='set parameter I to value V', metavar=('I', 'V') )
opts=parser.parse_args()

"""Init configuration"""
cfg = NestedDict()
cfg.bundle = 'reactor_anu_spectra_v01'
cfg.isotopes = [ 'U5', 'U8', 'Pu9', 'Pu1' ]
cfg.filename = ['data/Huber_smooth_extrap_{iso}_13MeV0.01MeVbin.dat', 'data/Mueller_smooth_extrap_{iso}_13MeV0.01MeVbin.dat']
cfg.strategy = dict( underflow='constant', overflow='extrapolate' )
cfg.edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) )
cfg.varname = 'avganushape.n{index:02d}'
# cfg.varmode = 'plain'
cfg.varmode = 'log'

"""Init inputs"""
points = N.linspace( 0.0, 12.0, 241 )
points_t = C.Points(points)
shared=NestedDict( points=points_t.single() )

ns = env.globalns('testexp')

"""Execute bundle"""
b, = execute_bundle( cfg=cfg, common_namespace=ns, shared=shared )

env.globalns.printparameters()

"""Plot result"""
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( L.u('anu_yield') )
ax.set_title( '' )

ax.vlines(cfg.edges, 0.0, 2.5, linestyles='--', alpha=0.5, colors='blue')

for name, output in b.outputs.items():
    ax.plot( points, output.data().copy(), label=L.s(name) )

if opts.set:
    for var, value in opts.set:
        par=ns('avganushape')[var]
        par.set(float(value))

    env.globalns.printparameters()

    for name, output in b.outputs.items():
        ax.plot( points, output.data().copy(), '--', label=L.s(name) )

if opts.log:
    ax.set_yscale('log')

ax.legend( loc='upper right' )

P.show()
