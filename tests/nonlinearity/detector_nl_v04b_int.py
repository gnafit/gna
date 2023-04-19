#!/usr/bin/env python

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

plt.rc('axes', grid=True)
plt.rc('lines', markerfacecolor='none')
plt.rc('xtick.minor', visible=True)
plt.rc('ytick.minor', visible=True)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-o', '--output', help='output file' )
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-f', '--flat', action='store_true', help='use flat spectrum' )
parser.add_argument( '-r', '--rect', action='store_true', help='use rectangular integrator' )
parser.add_argument( '-m', '--mapping', action='store_true', help='show mapping' )
parser.add_argument( '--supersample', '--ss', type=int, default=1, help='supersample N times', metavar='N' )
parser.add_argument('-x', '--xlim', nargs=2, type=float)
parser.add_argument('-y', '--ylim', nargs=2, type=float)
opts=parser.parse_args()

def hist_errorbar(edges, y, yerr=None, *args, **kwargs):
    centers=(edges[1:]+edges[:-1])*0.5
    hwidths=(edges[1:]-edges[:-1])*0.5

    kwargs.setdefault('fmt', 'none')

    return plt.errorbar(centers, y, yerr, hwidths, *args, **kwargs)

#
# Make input histogram
#
def singularities(values, edges):
    offset = (edges[1]-edges[0])*0.5
    values = values+offset
    indices = np.digitize(values, edges)-1
    indices = indices[indices<edges.size-1]
    phist = np.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

emin, emax = 0.0, 10.0
nbins = 2000 # 5 keV
# emax, nbins = 2.0, 400
# emin, emax, nbins = 0.8, 1.2, 800
edges = np.linspace(emin, emax, nbins+1, dtype='d')
binwidth = edges[1] - edges[0]
points = C.Points(edges, labels='Bin edges')
phist = singularities( [0.97, 1.00, 1.05, 1.10, 1.20, 2.20, 4.00, 7.00, 9.00], edges )
if opts.flat:
    phist[:]=0
    phist[int(nbins*(1.02-emin)/(emax-emin)):] = 1.0
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
            bundle = dict(name='energy_nonlinearity_db_root_subst', version='v02',
                major='l',
                ),
            # file to read
            filename = 'data/data_dayabay/tmp/detector_nl_consModel_450itr.root',
            # TGraph names. First curve will be used as nominal
            names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
                # The uncorrelated energy scale uncertainty type (absolute/relative)
            # par = uncertain(1.0, 0.2, 'percent'),
            # parname = 'escale',
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
def plot_projections(ax=None, *, update=False, relative=False, label='', plot_inverse=True, plot_int=True):
    def drawline():
        if relative:
            ax.axhline(1.0, linestyle='dashed', linewidth=0.5)
            ax.set_ylim(0.9,  1.15, auto=True)
        else:
            ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5)

    if ax:
        plt.sca(ax)
        drawline()
    else:
        newfigure = not update
        if newfigure:
            fig = plt.figure('proj')
            fig.clf()
            ax  = plt.subplot(111, xlabel='E', ylabel='E\'', title='')
            ax.set_xlim(0.5, edges[-1])

            drawline()
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

    mask = edges>0.499
    c = ax.plot(edges[mask], direct[mask], '-.',   label='Direct'+label, alpha=0.5)[0].get_color()
    if plot_inverse:
        mask=inverse>0.499
        ax.plot(inverse[mask], edges1[mask], ':',  label='Inverse'+label, alpha=0.5, color=c)
    if plot_int:
        ax.plot(evis, eq, '-.', label='Inverse, integration points'+label, alpha=0.5, color=c)

    ax.legend(ncol=2)

    return ax

def plotgradient(ax=None, update=False, figname='gradient'):
    if ax:
        plt.sca(ax)
    else:
        newfigure = not update
        if newfigure:
            fig = plt.figure(figname)
            fig.clf()
            ax  = plt.subplot(111, xlabel='E', ylabel='dE\'/dE', title='Gradient')
            # ax.set_xlim(0.5, edges[-1])
            # ax.set_ylim(1.0, 1.08)

            # ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5)
        else:
            fig = plt.figure(figname)
            ax=fig.gca()

    grad = context.outputs['lsnl_gradient'].data().copy()
    edges1 = context.outputs['lsnl_x'].data().copy()

    evis = context.outputs['lsnl_evis'].data().copy()
    deqdevis = context.outputs['lsnl_interpolator_grad'].data().copy()

    c=ax.plot(edges1, grad, '-.', label='Gradient')[0].get_color()
    ax.plot(evis, deqdevis, ':', label='Gradient interpolated', color=c)

    ax.legend()

    return ax

def plotmat(ax=None, transpose=False):
    ylabel = 'E corrected'
    xlabel = 'E true'
    if transpose:
        xlabel, ylabel = ylabel, xlabel
    if ax:
        plt.sca(ax)
        ax.set_ylabel(ylabel)
        def resetaspect():
            ax.set_aspect('auto')
        colorbar=False
    else:
        fig = plt.figure()
        ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title='')
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[-1], edges[0])
        def resetaspect(): pass
        colorbar=True

    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], '--', linewidth=0.5, color='black')

    mat = context.outputs['lsnl_matrix']
    mat.plot_matshow(mask=0.0, extent=[edges[0], edges[-1], edges[-1], edges[0]], colorbar=colorbar, transpose=transpose)
    resetaspect()

    direct = context.outputs['lsnl_direct'].data().copy()
    x, y = edges, direct
    if transpose:
        x, y = y, x
    ax.plot(x, y, '--', color='red', alpha=0.5)

    sm = mat.data().sum(axis=0)
    print('Mat sum:', sm)

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

def plothist(ax=None, ax1=None):
    if ax and ax1:
        plt.sca(ax)
    else:
        fig = plt.figure()
        gs = fig.add_gridspec(3, 1, hspace=0)
        ax1 = fig.add_subplot(gs[2], xlabel='E', ylabel='ratio', title='')

        ax = fig.add_subplot(gs[:2], xlabel='', ylabel='Entries', title='', sharex=ax1)
        ax.xaxis.set_tick_params(labelbottom=False)

    if opts.xlim: ax.set_xlim(*opts.xlim)
    if opts.ylim: ax.set_ylim(*opts.ylim)

    hist_evis = context.outputs['integral_evis']
    lsnl      = context.outputs['lsnl']
    hist_eq   = context.outputs['integral_eq']
    edges = context.outputs['energy_edges']
    hmax = hist_evis.data().max()

    energy    = context.outputs['energy'].data().copy()
    fcn_evis  = context.outputs['fcn_evis'].data().copy()
    fcn_evis[fcn_evis>0] = hmax

    energy1   = context.outputs['lsnl_evis'].data().copy()
    fcn_eq  = context.outputs['fcn_eq'].data().copy()
    fcn_eq*=hmax/fcn_eq.max()

    edges2 = context.outputs['lsnl_x'].data().copy()
    step_kev=(edges2[100]-edges2[99])*1000
    hist_evis.plot_hist(label='Original')
    lsnl.plot_hist(linestyle='--',    alpha=0.5, label=f'Smeared, ΔE={step_kev:.2f} keV')
    hist_eq.plot_hist(linestyle='-.', alpha=0.5, label='Variable substitution')

    plt.sca(ax1)
    top = lsnl.data()
    bottom = hist_eq.data()
    ratio = top/bottom
    ratio = np.ma.array(ratio, mask=(bottom==0))
    hist_errorbar(edges(), ratio, fmt='o', markerfacecolor='none', markersize=2)

    print('Modified:', lsnl.data)
    print('Substitution:', hist_eq.data())
    print('Original histogram:', hist_evis.data().sum())
    print('Modified histogram:', lsnl.data().sum())
    print('Subsitution histogram:', hist_eq.data().sum())

    # ax.plot(energy, fcn_evis, '.', color='black', markersize=1.5, label='Evis integration points (not to scale)')
    # ax.plot(energy1, fcn_eq, '.', color='red', markersize=1.5, label='Eq integration points (not to scale)')
    # ax.plot(energy, fcn_eq, '.', color='red', markersize=1.5, label='Eq integration points (not to scale)')

    ax.legend(loc='upper right', ncol=1)

def plot_all():
    fig = plt.figure(figsize=(6.4, 10))
    gs = fig.add_gridspec(7, 1, hspace=0)

    axratio = fig.add_subplot(gs[6], xlabel='E', ylabel='ratio')
    axbottom=axratio
    axhist = fig.add_subplot(gs[4:6], xlabel='', ylabel='height', sharex=axbottom)
    axquench = fig.add_subplot(gs[2], xlabel='', ylabel='rel. correction', sharex=axbottom)

    axmat = fig.add_subplot(gs[0:2], xlabel='', ylabel='', sharex=axbottom)
    axgrad = fig.add_subplot(gs[3], xlabel='', ylabel='grad', sharex=axbottom)

    for ax in (axmat, axhist, axquench, axratio, axgrad):
        if ax is not axratio:
            ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_ylim(auto=True)

    plotmat(axmat, transpose=False)
    plothist(axhist, axratio)
    plot_projections(axquench, relative=True, plot_inverse=True, plot_int=False)
    plotgradient(axgrad, figname='grad1')

def plot_mapping():
    energy = context.outputs['energy_edges'].data().copy()
    energy_ = context.outputs['lsnl_direct'].data().copy()

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='E, MeV', title='Mapping')
    ax.set_xticks(energy, minor=True)
    ax.grid(True, which='both')

    for e, e_ in zip(energy, energy_):
        ax.plot([e, e_], [1.0, 0.0], '-', linewidth=2.0, alpha=0.6)

#
# Plot commands
#
plot_projections()
savefig(opts.output, suffix='_projections')

plot_projections(relative=True)
savefig(opts.output, suffix='_projections_rel')

plotgradient(figname='grad0')
savefig(opts.output, suffix='_grad')

plotmat()
savefig(opts.output, suffix='_matrix')

plothist()
savefig(opts.output, suffix='_hist')

plot_all()

if opts.mapping:
    plot_mapping()

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


