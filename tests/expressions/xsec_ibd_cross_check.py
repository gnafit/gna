#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
#  from gna.expression import *
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna.bundle import execute_bundles
from load import ROOT as R
from gna import constructors as C
from gna.env import env
from matplotlib import pyplot as P
import numpy as np
from mpl_tools import bindings
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
R.GNAObject
#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument('-d', '--differential_xsec', action='store_true',
                    help='Plot differential cross section')
parser.add_argument('-i', '--dyboscar-input', help='Path to inputs from dyboscar')

args = parser.parse_args()

input_dyboscar = None
input_dtype=[('enu', 'd'), ('xsec0', 'd'), ('xsec1', 'd'), ('xsec1_c0', 'd'), ('xsec1_c1', 'd'), ('jac_c0', 'd'), ('jac_c1', 'd')]
if args.dyboscar_input:
    input_dyboscar = np.loadtxt(args.dyboscar_input, dtype=input_dtype)

dyboscar_ctheta = [0., 1.]

Evis_edges = C.Points(np.linspace(0.0, 12.0, 241, dtype='d'))
ctheta = C.Points(np.array([-1., -0.5, 0.0, 0.5, 1.0]))

indices = []
lib = dict()
expr =[
        'enu| ee| evis()',
        'ibd_zero_order(ee())',

        'enu_first_order| ee_first_order(evis()), ctheta()',
        'ibd_first_order| enu_first_order(), ctheta()',
        'jacobian(enu_first_order(), ee_first_order(), ctheta())'
        ]


a = Expression_v01(expr, indices)
print(a.expressions_raw)
print(a.expressions)
a.parse()
a.guessname(lib, save=True)

cfg = NestedDict(
        Evis = NestedDict(
            bundle = dict(name='predefined', version='v01'),
            name = 'evis',
            inputs=None,
            outputs = Evis_edges.single()
            ),
        ctheta = NestedDict(
            bundle = dict(name='predefined', version='v01'),
            name = 'ctheta',
            inputs=None,
            outputs = ctheta.single()
            ),
        #  kinint = NestedDict(
            #  bundle   = dict(name='integral_1d', version='v02'),
            #  variable = 'evis',
            #  edges    = np.linspace(0.0, 12.0, 241, dtype='d'),
            #  orders   = 3,
            #  provides = [ 'evis' ]
            #  ),
        #  kinint2 = NestedDict(
            #  bundle = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2')),
            #  variables = ('evis_fo', 'ctheta'),
            #  edges    = np.linspace(0.0, 12.0, 241, dtype='d'),
            #  xorders   = 7,
            #  yorder   = 21,
            #  provides = ['evis_fo', 'ctheta']
            #  ),
        first_order = NestedDict(
            bundle = dict(name='xsec_ibd', version='v02',
                          names=dict(ibd_xsec='ibd_first_order',
                                     enu='enu_first_order', ee='ee_first_order')),
            order = 1,
            provides = [ 'ibd_first_order', 'enu_first_order', 'ee_first_order', 'jacobian' ]
            ),
        zero_order = NestedDict(
            bundle = dict(name='xsec_ibd', version='v02', names=dict(ibd_xsec='ibd_zero_order')),
            order = 0,
            provides = [ 'ibd_zero_order', 'ee', 'enu' ]
            ),
        )
context = ExpressionContext_v01(cfg, ns=env.globalns)
a.build(context)

from gna.bindings import OutputDescriptor
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

#
# Do some plots
#
#  Initialize figure
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()

ax.set_xlabel(r'$E_\nu$, MeV')
ax.set_ylabel( r'$\sigma$' )
ax.set_title(r"IBD cross section vs $E_\nu$ in zero order")

# Get evis and cross section
evis = context.outputs.evis.data()
enu = context.outputs.enu.data()
xsec_zero_order = context.outputs.ibd_zero_order.data()

ax.plot(enu, xsec_zero_order, label='GNA')
if input_dyboscar is not None:
    ax.plot(input_dyboscar['enu'], input_dyboscar['xsec0'], label='dyboscar')

ax.set_xlim(1.8, 12.)
ax.legend(loc='upper left')

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel(r'$E_\nu$, MeV')
ax.set_ylabel( r'$\sigma$' )
ax.set_title(r"Differential IBD cross section vs $E_\nu$ in first order")

enu_first = context.outputs.enu_first_order.data().T
ctheta = context.outputs.ctheta.data()
xsec_first_order = context.outputs.ibd_first_order.data().T

for enu, cos, xsec in zip(enu_first, ctheta, xsec_first_order):
    if cos in dyboscar_ctheta:
        ax.plot(enu, xsec, label=r'GNA $\cos\,\theta = {}$'.format(cos))
if input_dyboscar is not None:
    ax.plot(input_dyboscar['enu'], input_dyboscar['xsec1_c0'], label=r'dybOscar $\cos\,\theta$ = 0')
    ax.plot(input_dyboscar['enu'], input_dyboscar['xsec1_c1'], label=r'dybOscar $\cos\,\theta$ = 1')

ax.set_xlim(1.8, 12.)
ax.legend(loc='best')


fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel(r'$E_\nu$, MeV')
ax.set_ylabel( r'Jacobian' )
ax.set_title("Jacobian of transition")

enu_first = context.outputs.enu_first_order.data().T
ctheta = context.outputs.ctheta.data()
jacobians = context.outputs.jacobian.data().T
ee_first = context.outputs.ee_first_order.data()

for enu, cos, jac in zip(enu_first, ctheta, jacobians):
    if cos in dyboscar_ctheta:
        ax.plot(ee_first, jac, label=r'GNA $\cos\,\theta = {}$'.format(cos))
if input_dyboscar is not None:
    ax.plot(input_dyboscar['enu'], input_dyboscar['jac_c0'], label=r'dybOscar $\cos\,\theta$ = 0')
    ax.plot(input_dyboscar['enu'], input_dyboscar['jac_c1'], label=r'dybOscar $\cos\,\theta$ = 0')
#  ax.set_xlim(1.8, 12.)
ax.set_ylim(0.97, 1.03)

ax.legend(loc='best')

if args.show:
    P.show()
