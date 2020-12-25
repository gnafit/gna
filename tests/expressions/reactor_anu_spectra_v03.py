#!/usr/bin/env python

"""4 antineutrino spectra for 4 fissile isotopes in the reactor

 Usage:
   1) Show the figure and save the diagram:
     ./tests/expressions/reactor_anu_spectra_v02.py -s --dot output/anuspectra_v02.dot
   2) Show the diagram:
     xdot output/anuspectra_v02.dot
 """

from load import ROOT as R
from gna.expression.expression_v01 import Expression_v01 as Expression, ExpressionContext_v01 as ExpressionContext
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

enu = C.Points(N.arange(1.0, 12.0+1.e-9, 0.001))

print()
cfg = NestedDict(
        enu = NestedDict(
            bundle = NestedDict(name='predefined', version='v01', major=''),
            name = 'enu',
            inputs = None,
            outputs = enu.single(),
            ),
        anuspec = NestedDict(
            bundle = dict(name='reactor_anu_spectra_v03', major='i'),
            name = 'anuspec',
            filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                        'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],

            edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
            ),
        )

"""Build"""
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )
print()

print( 'inputs:' )
print( context.inputs )

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

ax.vlines(cfg.anuspec.edges, 0.0, 2.5, linestyles='--', alpha=0.5, colors='blue')

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
        graph = GNADot(context.outputs.anuspec.U235, **kwargs)
        graph.write(opts.dot)
        print( 'Write output to:', opts.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

savefig(opts.output)

if opts.show:
    P.show()
