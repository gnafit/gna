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
from matplotlib.backends.backend_pdf import PdfPages
from os.path import abspath
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
R.GNAObject
#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-o', '--output', help='path to dump figures')
parser.add_argument('-d', '--differential_xsec', action='store_true',
                    help='Plot differential cross section')
parser.add_argument('-i', '--dyboscar-input', help='Path to inputs from dyboscar')

args = parser.parse_args()

input_dyboscar = None
input_dtype=[('enu', 'd'), ('xsec0', 'd'), ('xsec1', 'd'), ('xsec1_c0', 'd'), ('xsec1_c1', 'd'), ('jac_c0', 'd'), ('jac_c1', 'd')]
if args.dyboscar_input:
    input_dyboscar = np.loadtxt(args.dyboscar_input, dtype=input_dtype)

dyboscar_ctheta = [0., 1.]
__enu = np.linspace(0.0, 12.0, 1201, dtype='d')

Enue = C.Points(__enu)
Enue_first = C.Points(np.stack((__enu, __enu), axis=-1))
ctheta = C.Points(np.array([0.0, 1.]))
ns = env.ns('ibd')
ns.printparameters()

from gna.parameters.ibd import reqparameters

reqparameters(ns, pdg_year='dyboscar')

with ns:
    econv = R.EvisToEe()
    ibd =   R.IbdZeroOrder(True)
    ibd_first = R.IbdFirstOrder()


Enue.points.points >> ibd.xsec.Ee
Enue.points.points >> econv.Ee.Evis

Enue_first.points.points >> ibd_first.xsec.Enu
ctheta.points.points >> ibd_first.xsec.ctheta

Enue.points.points >> ibd_first.Enu.Ee
ctheta.points.points >> ibd_first.Enu.ctheta

Enue.points.points >> ibd_first.jacobian.Ee
ibd_first.Enu.Enu >> ibd_first.jacobian.Enu
ctheta.points.points >> ibd_first.jacobian.ctheta

Evis_edges = C.Points(np.linspace(0.0, 12.0, 1201, dtype='d'))

#
# Do some plots
#

if args.output:
    pp = PdfPages(abspath(args.output))
#  Initialize figure
fig, (ax, ax_ratio) = P.subplots(nrows=2, ncols=1 )
ax.minorticks_on()
ax.grid()

ax.set_xlabel(r'$E_\nu$, MeV')
ax.set_ylabel( r'$\sigma$' )
ax.set_title(r"IBD cross section vs $E_\nu$ in zero order")

ax_ratio.set_xlabel(r'$E_\nu$, MeV')
ax_ratio.set_ylabel(r'Ratio')
# Get evis and cross section
enue = __enu
xsec_zero_order = ibd.xsec.xsec.data()

ax.plot(enue, 1e41*xsec_zero_order, label='GNA')
if input_dyboscar is not None:
    assert(np.allclose(enue, input_dyboscar['enu']))
    ax.plot(input_dyboscar['enu'], 1e41*input_dyboscar['xsec0'], label='dyboscar')

ax.set_xlim(1.8, 12.)
ax.legend(loc='upper left')

if input_dyboscar is not None:
    ratio = xsec_zero_order/input_dyboscar['xsec0']

    ax_ratio.plot(input_dyboscar['enu'], xsec_zero_order / input_dyboscar['xsec0'], label='gna/dyboscar')
    ax_ratio.legend(loc='upper left')
    ax_ratio.set_ylim(0.9, 3)

if args.output:
    pp.savefig(fig)

fig, (ax, ax_ratio) = P.subplots(nrows=2, ncols=1)
ax.minorticks_on()
ax.grid()
ax.set_xlabel(r'$E_\nu$, MeV')
ax.set_ylabel( r'$\sigma$' )
ax.set_title(r"Differential IBD cross section vs $E_\nu$ in first order")

ax_ratio.set_xlabel(r'$E_\nu$, MeV')
ax_ratio.set_ylabel(r'Ratio')

enue_first = Enue_first.data()

xsec_first_order = ibd_first.xsec.data().T

ax.plot(enue, xsec_first_order[0], label=r'GNA $\cos\,\theta = {}$'.format(0.))
ax.plot(enue, xsec_first_order[1], label=r'GNA $\cos\,\theta = {}$'.format(1.))
if input_dyboscar is not None:
    ax.plot(input_dyboscar['enu'], input_dyboscar['xsec1_c0'], label=r'dybOscar $\cos\,\theta$ = 0', alpha=0.5)
    ax.plot(input_dyboscar['enu'], input_dyboscar['xsec1_c1'], label=r'dybOscar $\cos\,\theta$ = 1', alpha=0.5)

ax.set_xlim(1.8, 12.)
ax.legend(loc='best')

if input_dyboscar is not None:
    ax_ratio.plot(input_dyboscar['enu'], xsec_first_order[0]/input_dyboscar['xsec1_c0'], label=r'$\cos\,\theta = {}$'.format(0.))
    ax_ratio.plot(input_dyboscar['enu'], xsec_first_order[1]/input_dyboscar['xsec1_c1'], label=r'$\cos\,\theta = {}$'.format(1.))

ax_ratio.set_xlim(1.8, 12.)
ax_ratio.legend(loc='best')

fig, (ax, ax_ratio) = P.subplots(nrows=2, ncols=1  )
ax.minorticks_on()
ax.grid()
ax.set_xlabel(r'$E_\nu$, MeV')
ax.set_ylabel( r'Jacobian' )
ax.set_title("Jacobian of transition")

ee_first = Enue.single().data()
jacobians = ibd_first.jacobian.jacobian.data().T

ax.plot(ee_first, jacobians[0], label=r'GNA $\cos\,\theta = {0}$')
ax.plot(ee_first, jacobians[1], label=r'GNA $\cos\,\theta = {1}$')
if input_dyboscar is not None:
    ax.plot(input_dyboscar['enu'], input_dyboscar['jac_c0'], label=r'dybOscar $\cos\,\theta$ = 0')
    ax.plot(input_dyboscar['enu'], input_dyboscar['jac_c1'], label=r'dybOscar $\cos\,\theta$ = 1')

ax.set_ylim(0.97, 1.03)
ax.legend(loc='best')

if input_dyboscar is not None:
    ratio_c0 = jacobians[0] / input_dyboscar['jac_c0']
    ratio_c1 = jacobians[1] / input_dyboscar['jac_c1']

    ax_ratio.plot(ee_first, ratio_c0, label=r'$\cos\, \theta$ = 0')
    ax_ratio.plot(ee_first, ratio_c1, label=r'$\cos\, \theta$ = 1')
    ax_ratio.legend(loc='best')
    ax_ratio.set_xlim((0.3, 12))
    ax_ratio.set_ylim((0.9, 1.05))

if args.output:
    pp.savefig(fig)
    pp.close()

if args.show:
    P.show()
