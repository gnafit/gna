#!/usr/bin/env python

from load import ROOT as R
R.GNAObject
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig
from gna.bindings import common
from gna.env import env
import gna.constructors as C
import numpy as np
from gna.configurator import NestedDict
from gna.expression.expression_v01 import *
from gna.graphviz import savegraph
from typing import Optional

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
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='lowest bin edge')
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

emin, emax = opts.threshold, 10.0
step = 0.005
nbins_in = int((emax-emin)/step)
nbins_out = nbins_in*3
# emax, nbins_in = 2.0, 400
# emin, emax, nbins_in = 0.8, 1.2, 800
edges_in = np.linspace(emin, emax, nbins_in+1, dtype='d')
edges_out = np.linspace(emin, emax, nbins_out+1, dtype='d')
binwidth_in = edges_in[1] - edges_in[0]
edges_in_points = C.Points(edges_in, labels='Bin edges_in')
phist_in = singularities( [0.97, 1.00, 1.05, 1.10, 1.20, 2.20, 4.00, 7.00, 9.00], edges_in )
if opts.flat:
    phist_in[:]=0
    phist_in[int(nbins_in*(1.02-emin)/(emax-emin)):] = 1.0
hist_in = C.Histogram(edges_in, phist_in, labels='Input hist')
hist_out = C.Histogram(edges_out, labels='LSNL output hist (edges)')

phist_f = C.Points(np.hstack((phist_in/binwidth_in, [0.0])))
fcn_evis = C.InterpConst(labels=('InSegment (fcn_evis)', 'Interp fcn_evis'))
fcn_evis.setXY(edges_in_points.points.points, phist_f.points.points)
fcn_eq = C.InterpConst(labels=('InSegment (fcn_eq)', 'Interp fcn_eq'))
fcn_eq.setXY(edges_in_points.points.points, phist_f.points.points)
fcn_evis.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)
fcn_eq.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

#
# Initialize expression
#
indices = [
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3']),
    ]

lib = dict()

formulas = [
        #
        # Input
        #
        'hist_evis = integral_evis| fcn_evis| energy()',
        'lsnl_edges| hist_evis',
        'histedges_in| hist_in()',
        'histedges_out| hist_out()',
        #
        # LSNL C
        #
        'lsnl_coarse   = sum[l]| lsnl_weight[l] * lsnl_component_y[l]()',
        'lsnl_gradient = sum[l]| lsnl_weight[l] * lsnl_component_grad[l]()',
        'lsnl_interpolator_grad| lsnl_gradient',
        'lsnl_interpolator| lsnl_x(), lsnl_coarse, energy_edges(), energy()',
        'lsnl| hist_evis',
        #
        # LSNL D
        #
        'lsnl_coarse_d   = sum[l]| lsnl_weight_d[l] * lsnl_component_y_d[l]()',
        'lsnl_interpolator_d| lsnl_x_d(), lsnl_coarse_d, energy_edges(), energy(), histedges_out_edges()',
        'lsnl_edges_d| hist_in(), hist_out()',
        'lsnl_d| hist_evis',
        #
        # Substitution
        #
        'hist_eq = integral_eq| fcn_eq(lsnl_evis())/lsnl_interpolator_grad()',
        #
        # Energy Resolution
        #
        'eres_pars',
        # Matrix C
        'eres_sigmarel_input_m_c| eres_sigmarel_m_c| lsnl_outedges()',
        'eres_matrix_m_c| hist_evis, lsnl_outedges()',
        'eres_m_c| lsnl()',
        # Matrix D
        'eres_sigmarel_input_m_d| eres_sigmarel_m_d| histedges_out_centers()',
        'eres_matrix_m_d| hist_evis, hist_out()',
        'eres_m_d| lsnl_d()',
        # Matrix B
        'rebin_edges| integral_evis(), lsnl_outedges()',
        'eres_sigmarel_s| histedges_in_centers()',
        'eres_sigmarel_input_m_b| eres_sigmarel_s()',
        'eres_matrix_m_b| hist_evis',
        'eres_m_b| rebin_lsnl_c| lsnl()',
        # Substitution
        'eres_sigmarel_input_s| eres_sigmarel_s()',
        'eres_matrix_s| hist_evis',
        'eres_s| hist_eq'
        ]

expr = Expression_v01(formulas, indices=indices)

#
# Configuration
#
cfg = NestedDict(
        edges_in = dict(
            bundle=dict(name='predefined_v01'),
            name='hist_in',
            inputs=None,
            outputs=hist_in.single(),
            ),
        edges_out = dict(
            bundle=dict(name='predefined_v01'),
            name='hist_out',
            inputs=None,
            outputs=hist_out.single(),
            ),
        input = dict(
            bundle=dict(name='predefined_v01'),
            name='input',
            inputs=None,
            outputs=hist_in.single(),
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
        nonlin_c = dict(
            bundle = dict(name='energy_nonlinearity_db_root_subst', version='v03',
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
            supersample=opts.supersample,
            matrix_method = 'c'
            ),
        nonlin_d = dict(
            bundle = dict(name='energy_nonlinearity_db_root_subst', version='v03', major='l', names='_d'),
            # file to read
            filename = 'data/data_dayabay/tmp/detector_nl_consModel_450itr.root',
            # TGraph names. First curve will be used as nominal
            names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
                # The uncorrelated energy scale uncertainty type (abso_inlute/relative)
            # par = uncertain(1.0, 0.2, 'percent'),
            # parname = 'escale',
            nonlin_range = (0.5, 12.),
            expose_matrix = True,
            supersample=opts.supersample,
            matrix_method = 'd'
            ),
        #
        # Integration
        #
        integrator = dict(
            bundle=dict(name='integral_1d', version='v03'),
            variable='energy',
            edges    = edges_in,
            orders   = 200,
            integrator = opts.rect and 'rect' or 'gl',
            instances = {
                'integral_eq': 'Quenched energy integral',
                'integral_evis': 'Visible energy integral',
                }
            ),
        #
        # Eres
        #
        eres_pars=dict(
            # pars: sigma_e/e = sqrt(a^2 + b^2/E + c^2/E^2),
            # a - non-uniformity
            # b - statistical term
            # c - noise
            bundle=dict(name="parameters", version="v07"),
            pars=f"data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/juno_eres_v3.yaml",
            depth=2,
        ),
        eres_sigmarel_m_c=dict(
            bundle=dict(name="energy_resolution_sigmarel_abc", version="v01", names='_m_c'),
            parameter="eres_pars",
            pars=("a_nonuniform", "b_stat", "c_noise"),
        ),
        eres_sigmarel_m_d=dict(
            bundle=dict(name="energy_resolution_sigmarel_abc", version="v01", names='_m_d'),
            parameter="eres_pars",
            pars=("a_nonuniform", "b_stat", "c_noise"),
        ),
        eres_sigmarel_s=dict(
            bundle=dict(name="energy_resolution_sigmarel_abc", version="v01", names='_s'),
            parameter="eres_pars",
            pars=("a_nonuniform", "b_stat", "c_noise"),
        ),
        eres_smearing_m_c=dict(
            bundle=dict(name="detector_eres_inputsigma", version="v03", major="", names='_m_c'),
            eres_mode='erf',
            dynamic_edges=True
        ),
        eres_smearing_m_b=dict(
            bundle=dict(name="detector_eres_inputsigma", version="v03", major="", names='_m_b'),
            eres_mode='erf-square',
        ),
        eres_smearing_m_d=dict(
            bundle=dict(name="detector_eres_inputsigma", version="v03", major="", names='_m_d'),
            eres_mode='erf',
        ),
        eres_smearing_s=dict(
            bundle=dict(name="detector_eres_inputsigma", version="v03", major="", names='_s'),
            eres_mode='erf-square',
        ),
        bins=dict(
            bundle=dict(name="trans_histedges", version="v02"),
            types=("centers", "edges"),
            instances={
                "histedges_in": "E (in)",
                "histedges_out": "E (out, D)",
            },
        ),
        rebin=dict(
             bundle = dict(name='rebin_input_v01'),
             rounding = 6,
             instances = {
                 'rebin_lsnl_c': 'Rebin LSNL C',
                 'rebin_lsnl_d': 'Rebin LSNL D',
             },
             permit_underflow = True,
             expose_matrix = True,
             dynamic_edges = True
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

outputs = context.outputs
savegraph(outputs.energy_edges, opts.dot)
env.globalns.printparameters(labels=True)
print('Inputs')
print(context.inputs)
print('Outputs:')
print(outputs)

# context.required_bundles['rebin_lsnl_c'].bundle.objects[0].printtransformations()
context.required_bundles['eres_m_c'].bundle.objects[0].printtransformations()

#
# Plot functions
#
def plot_projections(ax0: plt.Axes=None, *, relative=False, label='', plot_inverse=True, plot_int=True, different_color=False) -> plt.Axes:
    def drawline():
        if relative:
            ax.axhline(1.0, linestyle='dashed', linewidth=0.5)
            ax.set_ylim(0.9,  1.15)
        else:
            ax.plot([edges_in[0], edges_in[-1]], [edges_in[0], edges_in[-1]], '--', linewidth=0.5)

    if ax0:
        ax = ax0
        plt.sca(ax)
        drawline()
    else:
        fig = plt.figure()
        ax  = plt.subplot(111, xlabel='E', ylabel='E\'', title='')
        ax.set_xlim(0.5, edges_in[-1])

        drawline()

    direct = outputs['lsnl_direct'].data().copy()
    inverse = outputs['lsnl_inverse'].data().copy()

    evis = outputs['lsnl_evis'].data().copy()
    eq   = outputs['energy'].data().copy()

    if relative:
        direct/=edges_in
        direct[np.isnan(direct)]=0.0

        edges1 = edges_in/inverse
        edges1[np.isnan(edges1)]=0.0

        eq/=evis
    else:
        edges1=edges_in

    mask = edges_in>0.499
    len = 5
    dash1 = (0*len, (1*len, 2*len))
    dash2 = (1*len, (1*len, 2*len))
    dash3 = (2*len, (1*len, 2*len))
    c = ax.plot(edges_in[mask], direct[mask], ls=dash1,   label='Direct'+label, alpha=0.5)[0].get_color()
    if different_color:
        kwargs = {}
    else:
        kwargs = {'color': c}
    if plot_inverse:
        mask=inverse>0.499
        ax.plot(inverse[mask], edges1[mask], ls=dash2,  label='Inverse'+label, alpha=0.5, **kwargs)
    if plot_int:
        ax.plot(evis, eq, ls=dash3, label='Inverse, integration points'+label, alpha=0.5, **kwargs)

    ax.legend(ncol=2)

    return ax

def plotgradient(ax0: Optional[plt.Axes]=None, figname: str='gradient') -> plt.Axes:
    if ax0:
        plt.sca(ax0)
        ax = ax0
    else:
        fig = plt.figure(figname)
        ax = plt.subplot(111, xlabel='E', ylabel='dE\'/dE', title='Gradient')
        # ax.set_xlim(0.5, edges_in[-1])
        # ax.set_ylim(1.0, 1.08)

        # ax.plot([edges_in[0], edges_in[-1]], [edges_in[0], edges_in[-1]], '--', linewidth=0.5)

    grad = outputs['lsnl_gradient'].data().copy()
    edges1 = outputs['lsnl_x'].data().copy()

    evis = outputs['lsnl_evis'].data().copy()
    deqdevis = outputs['lsnl_interpolator_grad'].data().copy()

    c=ax.plot(edges1, grad, '-.', label='Gradient')[0].get_color()
    ax.plot(evis, deqdevis, ':', label='Gradient interpolated', color=c)

    ax.legend()

    return ax

def plotmat(mat: np.ndarray,
            edges_a: np.ndarray, edges_b: np.ndarray,
            ax: Optional[plt.Axes]=None,
            transpose: bool=False) -> None:
    edges_b = outputs['lsnl_outedges'].data()

    ylabel = f'E corrected, {edges_b.size-1} bins'
    xlabel = f'E true, {edges_a.size-1} bins'
    if transpose:
        xlabel, ylabel = ylabel, xlabel
    if ax:
        plt.sca(ax)
        ax.set_ylabel(ylabel)
        def resetaspect():
            ax.set_aspect('auto')
        colorbar=False
        ax.set_xlim(edges_a[0], edges_a[-1])
        ax.set_ylim(edges_a[-1], edges_a[0])
    else:
        plt.figure()
        ax: plt.Axes = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title='')
        ax.set_xlim(edges_a[0], edges_a[-1])
        ax.set_ylim(edges_a[-1], edges_a[0])
        def resetaspect(): pass
        colorbar=True

    ax.plot([edges_a[0], edges_a[-1]], [edges_a[0], edges_a[-1]], '--', linewidth=0.5, color='black')

    matdata = mat.data()
    if matdata.shape[0]==matdata.shape[1]:
        mat.plot_matshow(mask=0.0, extent=[edges_a[0], edges_a[-1], edges_a[-1], edges_a[0]], colorbar=colorbar, transpose=transpose)
    else:
        ax.pcolorfast(edges_a, edges_b, np.ma.array(matdata, mask=(matdata==0.0)))
    resetaspect()

    direct = outputs['lsnl_direct'].data().copy()
    x, y = edges_a, direct
    if transpose:
        x, y = y, x
    ax.plot(x, y, '--', color='red', alpha=0.5)

    sm = mat.data().sum(axis=0)
    print('Mat sum:', sm)

def checktaint():
    lsnl = outputs['rebin_lsnl_c']
    mat = outputs['lsnl_matrix']
    edges_in = outputs['energy_edges']
    direct = outputs['lsnl_direct']
    inverse = outputs['lsnl_inverse']

    lsnl.data()
    assert not edges_in.getTaintflag().tainted()
    assert not direct.getTaintflag().tainted()
    assert not inverse.getTaintflag().tainted()
    assert not mat.getTaintflag().tainted()
    assert not lsnl.getTaintflag().tainted()

    edges_in.getTaintflag().taint()
    assert edges_in.getTaintflag().tainted()
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

    hist_evis = outputs['integral_evis']
    lsnl_b    = outputs['rebin_lsnl_c']
    hist_eq   = outputs['integral_eq']
    hist_eres_matrix_b = outputs['eres_m_b']
    hist_eres_matrix_c = outputs['eres_m_c']
    hist_eres_matrix_d = outputs['eres_m_d']
    hist_eres_subst = outputs['eres_s']
    edges_in = outputs['energy_edges']
    hmax = hist_evis.data().max()

    energy    = outputs['energy'].data().copy()
    fcn_evis  = outputs['fcn_evis'].data().copy()
    fcn_evis[fcn_evis>0] = hmax

    energy1   = outputs['lsnl_evis'].data().copy()
    fcn_eq  = outputs['fcn_eq'].data().copy()
    fcn_eq*=hmax/fcn_eq.max()

    edges2 = outputs['lsnl_x'].data().copy()
    step_kev=(edges2[100]-edges2[99])*1000
    kwargs = {'alpha': 0.5, 'linewidth': 1}
    hist_evis.plot_hist(label='Original')
    lsnl_b.plot_hist(            linestyle='-', color='red',   label=f'Matrix B, ΔE={step_kev:.2f} keV', **kwargs)
    hist_eq.plot_hist(           linestyle='-', color='green', label='Substitution', **kwargs)
    hist_eres_matrix_b.plot_hist(linestyle='-', color='red',   label=f'Matrix B, smeared, ΔE={step_kev:.2f} keV', **kwargs)
    hist_eres_matrix_c.plot_hist(linestyle='-', color='blue',  label=f'Matrix C, smeared, ΔE={step_kev:.2f} keV', **kwargs)
    hist_eres_matrix_d.plot_hist(linestyle='-', color='gold',  label=f'Matrix D, smeared', **kwargs)
    hist_eres_subst.plot_hist(   linestyle='-', color='green', label=f'Subst, smeared', **kwargs)

    plt.sca(ax1)
    ax1.set_ylim(-0.01, 0.01)
    top = lsnl_b.data()
    bottom = hist_eq.data()
    ratio = np.log(top/bottom)
    ratio = np.ma.array(ratio, mask=(bottom==0))
    hist_errorbar(edges_in(), ratio, fmt='o', markerfacecolor='none', markersize=2, color='red', label='matrix B/subst (no eres)')

    hist_eres_matrix_b.plot_hist(logratio=hist_eres_subst, color='red',  label='matrix B/subst')
    hist_eres_matrix_c.plot_hist(logratio=hist_eres_subst, color='blue', label='matrix C/subst')
    hist_eres_matrix_d.plot_hist(logratio=hist_eres_subst, color='gold', label='matrix D/subst')

    print('Modified:', lsnl_b.data)
    print('Substitution:', hist_eq.data())
    print('Original histogram:', hist_evis.data().sum())
    print('Modified histogram:', lsnl_b.data().sum())
    print('Subsitution histogram:', hist_eq.data().sum())

    # ax.plot(energy, fcn_evis, '.', color='black', markersize=1.5, label='Evis integration points (not to scale)')
    # ax.plot(energy1, fcn_eq, '.', color='red', markersize=1.5, label='Eq integration points (not to scale)')
    # ax.plot(energy, fcn_eq, '.', color='red', markersize=1.5, label='Eq integration points (not to scale)')

    ax.legend(loc='upper right', fontsize='small')

    fig = plt.gcf()
    fig.canvas.draw()
    ot = ax1.get_xaxis().get_offset_text()
    ot.set_position((0,1))
    fig.canvas.draw()

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

    plotmat(outputs['lsnl_matrix'], edges_in, edges_out, axmat, transpose=False)
    plothist(axhist, axratio)
    plot_projections(axquench, relative=True, plot_inverse=True, plot_int=False)
    plotgradient(axgrad, figname='grad1')

def plot_mapping():
    energy = outputs['energy_edges'].data().copy()
    energy_ = outputs['lsnl_direct'].data().copy()

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='E, MeV', title='Mapping')
    ax.set_xticks(energy, minor=True)
    ax.grid(True, which='both')

    for e, e_ in zip(energy, energy_):
        ax.plot([e, e_], [1.0, 0.0], '-', linewidth=2.0, alpha=0.6)

def plot_edges():
    energy_out  = outputs['lsnl_outedges'].data().copy()
    energy_in = outputs['energy_edges'].data().copy()

    de_out = energy_out[1:] - energy_out[:-1]
    de_in = energy_in[1:] - energy_in[:-1]

    de_out_limited = de_out[np.fabs(de_out)<2*step]
    de_max = max(de_out_limited.max(), de_in.max())
    de_min = min(de_out_limited.min(), de_in.min())

    plt.figure()
    ax = plt.subplot(111, xlabel='E, MeV', ylabel='de, MeV', title='Binning')
    # ax.set_xticks(energy, minor=True)
    ax.grid(True, which='both')

    ax.set_xlim(energy_in[0], energy_in[-1])
    ax.set_ylim(min(-0.001, de_min*1.1), de_max*1.1)

    n_in_limits = (energy_out<=energy_in[-1]).sum()
    n_out_limits = (energy_out>energy_in[-1]).sum()
    frac = n_in_limits/energy_in.size*100

    ax.plot(energy_out[:-1], de_out, 'o', label=f'edges out: {n_in_limits} ({frac:.0f}%)|{n_out_limits}')
    ax.plot(energy_in[:-1], de_in, 'o', label='edges in')

    ax.legend()

#
# Plot commands
#
plot_projections(different_color=True)
savefig(opts.output, suffix='_projections')

plot_projections(relative=True, different_color=True)
savefig(opts.output, suffix='_projections_rel')

plotgradient(figname='grad0')
savefig(opts.output, suffix='_grad')

plotmat(outputs['lsnl_matrix'], edges_in, edges_out)
savefig(opts.output, suffix='_matrix')

# plothist()
savefig(opts.output, suffix='_hist')

# plot_all()

if opts.mapping:
    plot_mapping()

# Check impact of the parameters
pars = env.globalns('lsnl_weight').items()

ax = plot_projections(relative=True)
axg = plotgradient()
for name, par in pars:
    if name=='nominal': continue
    par.set(1.0)

    axp=plot_projections(ax, label=' '+name, relative='True')
    axg=plotgradient(axg)

    par.set(0.0)

plt.sca(axp)
savefig(opts.output, suffix='_curves_rel')
plt.sca(axg)
savefig(opts.output, suffix='_gradients')

plot_edges()

# checktaint()

if opts.show:
    plt.show()


