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
import numpy as np
from gna.configurator import NestedDict, uncertain
from gna.expression.expression_v01 import *
from gna.graphviz import savegraph
import scipy

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-o', '--output', help='output file' )
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-f', '--flat', action='store_true', help='use flat spectrum' )
parser.add_argument( '-r', '--rect', action='store_true', help='use rectangular integrator' )
parser.add_argument( '--supersample', '--ss', type=int, help='supersample N times', metavar='N' )
opts=parser.parse_args()

#
# Make input histogram
#
def singularities( values, edges ):
    indices = np.digitize( values, edges )-1
    phist = np.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

emin, emax = 0.0, 10.0
nbins = 2000
edges = np.linspace(emin, emax, nbins+1, dtype='d')
binwidth = edges[1] - edges[0]
points = C.Points(edges, labels='Bin edges')
phist = singularities( [ 1.225, 2.225, 4.025, 7.025, 9.025 ], edges )
if opts.flat:
    phist[int(nbins*(1.0-emin)/(emax-emin)):] = 1.0
hist = C.Histogram(edges, phist, labels='Input hist')

phist_f = C.Points(np.hstack((phist/binwidth, [0.0])))
fcn_evis = C.InterpConst(labels=('InSegment (fcn_evis)', 'Interp fcn_evis'))
fcn_evis.setXY(points.points.points, phist_f.points.points)
fcn_eq = C.InterpConst(labels=('InSegment (fcn_eq)', 'Interp fcn_eq'))
fcn_eq.setXY(points.points.points, phist_f.points.points)
fcn_evis.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)
fcn_eq.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

#
# Initialize expression
#
indices = [
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'])
    ]

lib = dict()

formulas = [
        'lsnl_coarse   = sum[l]| lsnl_weight[l] * lsnl_component_y[l]()',
        'lsnl_gradient = sum[l]| lsnl_weight[l] * lsnl_component_grad[l]()',
        'lsnl_interpolator_grad| lsnl_gradient',
        'lsnl_interpolator| lsnl_x(), lsnl_coarse, energy_edges(), energy()',
        'hist_evis = integral_evis| fcn_evis| energy()',
        'lsnl_edges| hist_evis',
        'lsnl| hist_evis',
        'hist_eq = integral_eq| fcn_eq(lsnl_evis())/lsnl_interpolator_grad()',
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
        fcn_evis = dict(
            bundle=dict(name='predefined_v01'),
            name='fcn_evis',
            inputs=((fcn_evis.interp.newx, fcn_evis.insegment.points),),
            outputs=fcn_evis.interp.interp,
            ),
        fcn_eq = dict(
            bundle=dict(name='predefined_v01'),
            name='fcn_eq',
            inputs=((fcn_eq.interp.newx, fcn_eq.insegment.points),),
            outputs=fcn_eq.interp.interp,
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
            supersample=opts.supersample
            ),
        integrator = dict(
            bundle=dict(name='integral_1d', version='v03'),
            variable='energy',
            edges    = edges,
            orders   = 100,
            integrator = opts.rect and 'rect' or 'gl',
            instances = {
                'integral_eq': 'Quenched energy integral',
                'integral_evis': 'Visible energy integral',
                }
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
savegraph(context.outputs.energy_edges, opts.dot)
env.globalns.printparameters(labels=True)

#
# Plot functions
#
def plot_projections(update=False, relative=False, label=''):
    newfigure = not update
    if newfigure:
        fig = plt.figure('proj')
        fig.clf()
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
        fig = plt.figure('proj')
        ax=fig.gca()

    direct = context.outputs['lsnl_direct'].data().copy()
    inverse = context.outputs['lsnl_inverse'].data().copy()

    evis = context.outputs['lsnl_evis'].data().copy()
    eq   = context.outputs['energy'].data().copy()

    if relative:
        direct/=edges
        direct[np.isnan(direct)]=0.0

        edges1 = edges/inverse
        edges1[np.isnan(edges1)]=0.0

        eq/=evis
    else:
        edges1=edges

    c = ax.plot(edges, direct, '-.',   label='Direct'+label, alpha=0.5)[0].get_color()
    ax.plot(inverse, edges1, ':',  label='Inverse'+label, alpha=0.5, color=c)
    ax.plot(evis, eq, '-.', label='Inverse, integration points'+label, alpha=0.5, color=c)

    ax.legend()

    return ax

def plotgradient(update=False):
    newfigure = not update
    if newfigure:
        fig = plt.figure('gradient')
        fig.clf()
        ax  = plt.subplot(111, xlabel='E', ylabel='dE\'/dE', title='Gradient')
        ax.minorticks_on()
        ax.grid()
        ax.set_xlim(0.5, edges[-1])
        ax.set_ylim(0.7, 1.1)

        # ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5)
    else:
        fig = plt.figure('gradient')
        ax=fig.gca()

    grad = context.outputs['lsnl_gradient'].data().copy()
    edges1 = context.outputs['lsnl_x'].data().copy()

    evis = context.outputs['lsnl_evis'].data().copy()
    deqdevis = context.outputs['lsnl_interpolator_grad'].data().copy()

    c=ax.plot(edges1, grad, '-.', label='Gradient')[0].get_color()
    ax.plot(evis, deqdevis, ':', label='Gradient interpolated', color=c)

    ax.legend()

    return ax

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
    edges = context.outputs['energy_edges']
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

    hist_evis = context.outputs['integral_evis']
    lsnl      = context.outputs['lsnl']
    hist_eq   = context.outputs['integral_eq']
    hmax = hist_evis.data().max()

    energy    = context.outputs['energy'].data().copy()
    fcn_evis  = context.outputs['fcn_evis'].data().copy()
    fcn_evis[fcn_evis>0] = hmax

    energy1   = context.outputs['lsnl_evis'].data().copy()
    fcn_eq  = context.outputs['fcn_eq'].data().copy()
    fcn_eq*=hmax/fcn_eq.max()

    hist_evis.plot_hist(label='Original')
    lsnl.plot_hist(linestyle='--',    alpha=0.5, label='Smeared')
    hist_eq.plot_hist(linestyle='-.', alpha=0.5, label='Variable substitution')

    print('Modified:', lsnl.data)
    print('Substitution:', hist_eq.data())
    print('Original histogram:', hist_evis.data().sum())
    print('Modified histogram:', lsnl.data().sum())
    print('Subsitution histogram:', hist_eq.data().sum())

    ax.plot(energy, fcn_evis, '.', color='black', markersize=1.5, label='Evis integration points (not to scale)')
    # ax.plot(energy1, fcn_eq, '.', color='red', markersize=1.5, label='Eq integration points (not to scale)')
    ax.plot(energy, fcn_eq, '.', color='red', markersize=1.5, label='Eq integration points (not to scale)')

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
plotgradient()
for name, par in pars:
    if name=='nominal': continue
    par.set(1.0)

    axp=plot_projections(update=True, label=' '+name, relative='True')
    axg=plotgradient(update=True)

    par.set(0.0)

plt.sca(axp)
savefig(opts.output, suffix='_curves_rel')
plt.sca(axg)
savefig(opts.output, suffix='_gradients')

# checktaint()

if opts.show:
    plt.show()


