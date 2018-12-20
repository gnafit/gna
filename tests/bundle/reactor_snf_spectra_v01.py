Broken: subbundle removed

!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundles
from gna.env import env
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar
import gna.constructors as C
from gna.labelfmt import formatter as L

"""Parse arguments"""
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
# parser.add_argument( '--set', nargs=2, action='append', default=[], help='set parameter I to value V', metavar=('I', 'V') )
# parser.add_argument( '--rset', nargs=2, action='append', default=[], help='set parameter I to value central+sigma*V', metavar=('I', 'V') )
parser.add_argument( '--dot', help='write graphviz output' )
opts=parser.parse_args()

"""Init configuration"""
cfg = NestedDict()
cfg.bundle = ['reactor_fission_fractions_const_v01', 'subbundle:anu', 'subbundle:offeq', 'subbundle:snf']
cfg.isotopes = [ 'U5', 'U8', 'Pu9', 'Pu1' ]
cfg.reactors = [ 'DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4' ]
cfg.fission_fractions = NestedDict( # Nucl.Instrum.Meth.A569:837-844,2006
        [
            ('U235' , 0.563),
            ('U238' , 0.079),
            ('Pu239', 0.301),
            ('Pu241', 0.057)
        ] )

cfg.anu = NestedDict(
    bundle = 'reactor_anu_spectra_v01',
    isotopes = [ 'U5', 'U8', 'Pu9', 'Pu1' ],
    filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
    strategy = dict( underflow='constant', overflow='extrapolate' ),
    edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
)

cfg.snf = NestedDict(
        bundle='reactor_snf_spectra_v01',
        filename = 'data/reactor_anu_spectra/SNF/kopeikin_0412.044_spent_fuel_spectrum_smooth.dat',
        norm=uncertain( 1.0, 100, mode='percent' ),
        edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.25 ), [ 12.3 ] ) )
        )

cfg.offeq = NestedDict(
        bundle='reactor_offeq_spectra_v01',
        skip=['U238'],
        filename = 'data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
        norm=uncertain( 1.0, 30, mode='percent' ),
        edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.25 ), [ 12.3 ] ) )
        )

"""Init inputs"""
points = N.linspace( 0.0, 12.0, 241 )
points_t = C.Points(points)
points_t.points.setLabel('E (integr)')
shared=NestedDict( points=points_t.single() )

ns = env.globalns('testexp')

"""Execute bundle"""
bundles = execute_bundles( cfg=cfg, common_namespace=ns, shared=shared )
snf = bundles[-1]
offeq = bundles[-2]

env.globalns.printparameters( labels=True )

"""Plot result"""
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( '' )
ax.set_title( '' )

ax.plot( snf.edges, snf.ratio )

# ax.vlines(cfg.edges, 0.0, 2.5, linestyles='--', alpha=0.5, colors='blue')

# for name, output in b.outputs.items():
    # ax.plot( points, output.data().copy(), label=L.s(name) )

# if opts.set or opts.rset:
    # for var, value in opts.set:
        # par=ns[var]
        # par.set(float(value))
    # for var, value in opts.rset:
        # par=ns[var]
        # par.setNormalValue(float(value))

    # print()
    # print('Parameters after modification')
    # env.globalns.printparameters()

    # for name, output in b.outputs.items():
        # ax.plot( points, output.data().copy(), '--', label=L.s(name) )

# if opts.log:
    # ax.set_yscale('log')

# ax.legend( loc='upper right' )

if opts.dot:
    if True:
        from gna.graphviz import GNADot

        kwargs=dict(
                # splines='ortho'
                )
        ts = [ snf.transformations_out.values()[0] ]+list(offeq.transformations_out.values()[0].values())
        graph = GNADot( ts, **kwargs )
        graph.write(opts.dot)
        print( 'Write output to:', opts.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )

if opts.show:
    P.show()
