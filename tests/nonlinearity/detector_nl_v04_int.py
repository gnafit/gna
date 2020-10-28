#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundles
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.bindings import common
from gna.env import env
import gna.constructors as C
import numpy as N
from gna.configurator import NestedDict, uncertain
from gna.expression.expression_v01 import *
from gna.graphviz import savegraph

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-o', '--output', help='output file' )
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
opts=parser.parse_args()

#
# Make input histogram
#
def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

nbins = 240
edges = N.linspace(0.0, 12.0, nbins+1, dtype='d')
points = C.Points(edges, labels='Bin edges')
phist = singularities( [ 1.225, 2.225, 4.025, 7.025, 9.025 ], edges )
hist = C.Histogram(edges, phist, labels='Input hist')

#
# Initialize expression
#
indices = [
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'])
    ]

lib = dict()

formulas = [
        'evis_edges()',
        'lsnl_coarse   = sum[l]| lsnl_weight[l] * lsnl_component_y[l]()',
        'lsnl_gradient = sum[l]| lsnl_weight[l] * lsnl_component_grad[l]()',
        'lsnl_interpolator_grad| lsnl_gradient',
        'lsnl_interpolator| lsnl_x(), lsnl_coarse, evis_edges(), energy()',
        'lsnl_edges| hist()',
        'lsnl| hist()',
        'integral()'
        ]

expr = Expression_v01(formulas, indices=indices)

#
# Configuration
#
cfg = NestedDict(
        edges = dict(
            bundle=dict(name='predefined_v01'),
            name='hist',
            inputs=None,
            outputs=hist.single(),
            ),
        input = dict(
            bundle=dict(name='predefined_v01'),
            name='input',
            inputs=None,
            outputs=hist.single(),
            ),
        edges1 = dict(
            bundle=dict(name='predefined_v01'),
            name='evis_edges',
            inputs=None,
            outputs=points.single(),
            ),
        nonlin = dict(
            bundle = dict(name='energy_nonlinearity_db_root_subst', version='v01',
                major='l',
                ),
            # file to read
            filename = 'data/data_dayabay/tmp/detector_nl_consModel_450itr.root',
            # TGraph names. First curve will be used as nominal
            names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
                # The uncorrelated energy scale uncertainty type (absolute/relative)
            par = uncertain(1.0, 0.2, 'percent'),
            parname = 'escale',
            extrapolation_strategy = 'extrapolate',
            nonlin_range = (0.5, 12.),
            expose_matrix = True,
            ),
        integrator = dict(
            bundle=dict(name='integral_1d', version='v02'),
            variable='energy',
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            orders   = 5,
            )
    )

#
# Initialize bundles
#
expr.parse()
expr.guessname(lib, save=True)
expr.tree.dump()
context = ExpressionContext_v01(cfg, ns=env.globalns)
expr.build(context)

print(context.outputs)
savegraph(context.outputs.evis_edges, opts.dot)
env.globalns.printparameters(labels=True)

#
# Plot functions
#
def plot_projections(update=False, relative=False, label=''):
    newfigure = not update
    if newfigure:
        fig = plt.figure()
        ax  = plt.subplot(111, xlabel='E', ylabel='E\'', title='')
        ax.minorticks_on()
        ax.grid()
        ax.set_xlim(0.5, edges[-1])

        if relative:
            ax.axhline(1.0, linestyle='dashed', linewidth=0.5)
            ax.set_ylim(0.8, 1.2)
        else:
            ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5)
    else:
        ax=plt.gca()

    direct = context.outputs['lsnl_direct'].data().copy()
    inverse = context.outputs['lsnl_inverse'].data().copy()

    if relative:
        direct/=edges
        direct[N.isnan(direct)]=0.0

        edges1 = edges/inverse
        edges1[N.isnan(edges1)]=0.0
    else:
        edges1=edges

    ax.plot(edges, direct, '-.', label='Direct'+label, alpha=0.7)
    ax.plot(inverse, edges1, ':', label='Inverse'+label, alpha=0.7)

    ax.legend()

def plotgradient(update=False):
    newfigure = not update
    if newfigure:
        fig = plt.figure()
        ax  = plt.subplot(111, xlabel='E', ylabel='dE\'/dE', title='Gradient')
        ax.minorticks_on()
        ax.grid()
        ax.set_xlim(0.5, edges[-1])

        # ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5)
    else:
        ax=plt.gca()

    grad = context.outputs['lsnl_gradient'].data().copy()
    edges1 = context.outputs['lsnl_x'].data().copy()

    ax.plot(edges1, grad, '-.', label='Gradient')

    # ax.legend()

def plotmat():
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title='')
    ax.minorticks_on()
    ax.grid()
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[-1], edges[0])

    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5, color='black')

    mat = context.outputs['lsnl_matrix']
    mat.plot_matshow(mask=0.0, extent=[edges[0], edges[-1], edges[-1], edges[0]], colorbar=True)

    direct = context.outputs['lsnl_direct'].data().copy()
    ax.plot(edges, direct, '--', color='white')

    sm = mat.data().sum(axis=0)
    print(sm)

def checktaint():
    lsnl = context.outputs['lsnl']
    mat = context.outputs['lsnl_matrix']
    edges = context.outputs['evis_edges']
    direct = context.outputs['lsnl_direct']
    inverse = context.outputs['lsnl_inverse']

    lsnl.data()
    assert not edges.getTaintflag().tainted()
    assert not direct.getTaintflag().tainted()
    assert not inverse.getTaintflag().tainted()
    assert not mat.getTaintflag().tainted()
    assert not lsnl.getTaintflag().tainted()

    edges.getTaintflag().taint()
    assert edges.getTaintflag().tainted()
    assert direct.getTaintflag().tainted()
    assert inverse.getTaintflag().tainted()
    assert mat.getTaintflag().tainted()
    assert lsnl.getTaintflag().tainted()
    lsnl.data()

def plothist():
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title='')
    ax.minorticks_on()
    ax.grid()

    lsnl = context.outputs['lsnl']
    hist = context.outputs['hist']

    hist.plot_hist(label='Original')
    lsnl.plot_hist(label='Smeared')

    ax.legend()

#
# Plot commands
#
plot_projections()
savefig(opts.output, suffix='_projections')

plot_projections(relative=True)
savefig(opts.output, suffix='_projections_rel')

plotgradient()
savefig(opts.output, suffix='_grad')

plotmat()
savefig(opts.output, suffix='_matrix')

plothist()
savefig(opts.output, suffix='_hist')

# Check impact of the parameters
pars = env.globalns('lsnl_weight').items()

plot_projections(relative=True)
for name, par in pars:
    if name=='nominal': continue
    par.set(1.0)

    plot_projections(update=True, label=' '+name, relative='True')

    par.set(0.0)
savefig(opts.output, suffix='_curves_rel')

checktaint()

if opts.show:
    plt.show()


