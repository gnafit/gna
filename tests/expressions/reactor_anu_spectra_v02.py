#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""4 antineutrino spectra for 4 fissile isotopes in the reactor

 Usage:
   1) Show the figure and save the diagram:
     ./tests/expressions/reactor_anu_spectra_v02.py -s --dot output/anuspectra_v02.dot
   2) Show the diagram:
     xdot output/anuspectra_v02.dot
 """

from __future__ import print_function
from load import ROOT as R
from gna.expression import *
import numpy as N
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundles
from gna.env import env
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar, savefig
import gna.constructors as C
from gna.labelfmt import formatter as L

"""Parse arguments"""
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-o', '--output', help='output figure' )
parser.add_argument( '-l', '--log', action='store_true', help='logarithmic scale' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '--dot', help='write graphviz output' )
opts=parser.parse_args()

"""Init expression"""
indices = [
    ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241'])
    ]

lib = dict(
)

expr = 'anuspec[i]( enu() )'
a = Expression(expr, indices=indices)

print(a.expressions_raw)
print(a.expressions)

a.parse()
a.guessname(lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        enu = NestedDict(
            bundle = 'arange',
            name = 'enu',
            provides = ['enu'],
            args = ( 0.0, 12.01, 0.05 )
            ),
        anuspec = NestedDict(
            bundle = 'reactor_anu_spectra_v02',
            name = 'anuspec',
            filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                            'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],

            # strategy = dict( underflow='constant', overflow='extrapolate' ),
            edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),

            # corrections=NestedDict(
                    # bundle       = 'bundlelist_v01',
                    # bundles_list = [ 'free', 'uncorrelated', 'correlated' ],
                    # free = NestedDict(
                        # bundle  ='reactor_anu_freemodel_v01',
                        # varname = 'avganushape.n{index:02d}',
                        # varmode = 'log', # 'plain'
                        # ),
                    # uncorrelated = NestedDict(
                        # bundle        = 'reactor_anu_uncorr_v01',
                        # uncnames      = '{isotope}_uncorr.uncn{index:02d}',
                        # uncertainties = ['data/reactor_anu_spectra/Huber/reac_anu_uncertainties_huber_{isotope}_{mode}.dat',
                                         # 'data/reactor_anu_spectra/Mueller/reac_anu_uncertainties_mueller_{isotope}_{mode}.dat']
                        # ),
                    # correlated = NestedDict(
                        # bundle   = 'reactor_anu_corr_v01',
                        # uncname  = 'isotopes_corr',
                        # parnames = '{isotope}_corr.uncn{index:02d}',
                        # uncertainties = ['data/reactor_anu_spectra/Huber/reac_anu_uncertainties_huber_{isotope}_{mode}.dat',
                                         # 'data/reactor_anu_spectra/Mueller/reac_anu_uncertainties_mueller_{isotope}_{mode}.dat']
                        # )
                    # )
            ),
        )


"""Build"""
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

ns = env.globalns('testexp')
env.globalns.printparameters( labels=True )

"""Plot result"""
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( L.u('anu_yield') )
ax.set_title( '' )

# ax.vlines(cfg.edges, 0.0, 2.5, linestyles='--', alpha=0.5, colors='blue')

for name, output in context.outputs.anuspec.items():
    data=output.data().copy()
    ax.plot( context.outputs.enu.data(), N.ma.array(data, mask=data==0.0), label=L.s(name) )

if opts.log:
    ax.set_yscale('log')

ax.legend( loc='upper right' )

if opts.dot:
    try:
        from gna.graphviz import GNADot

        kwargs=dict(
                # splines='ortho'
                joints=False,
                )
        graph = GNADot(context.outputs.enu, **kwargs)
        graph.write(opts.dot)
        print( 'Write output to:', opts.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

savefig(opts.output)

if opts.show:
    P.show()
