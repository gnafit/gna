#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna import constructors as C
from gna.bindings import common
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig, add_to_labeled_items, plot_hist
from gna.graphviz import savegraph
from gna.env import env
from matplotlib.backends.backend_pdf import PdfPages
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict
from scipy.signal import argrelmin, argrelmax
import itertools as I
from argparse import ArgumentParser, Namespace

def merge_extrema(x_min, x_max, y_min, y_max):
    x_ext = N.zeros(x_min.size+x_max.size)
    y_ext = x_ext.copy()
    counter = 0
    for i, (a, b, ya, yb) in enumerate(I.izip_longest(x_min, x_max, y_min, y_max)):
        if b>a:
            if a is not None:
                x_ext[counter], y_ext[counter] = a, ya
                counter+=1
            if b is not None:
                x_ext[counter], y_ext[counter] = b, yb
                counter+=1
        else:
            if a is not None:
                x_ext[counter], y_ext[counter] = b, yb
                counter+=1
            if b is not None:
                x_ext[counter], y_ext[counter] = a, ya
                counter+=1

    return x_ext, y_ext

def merge_extrema(x_min, x_max, y_min, y_max):
    return x_min, y_min

class Data(Namespace):
    def __init__(self, dshape, extshape):
        self.psur = N.zeros(dshape)
        self.psur_ext_x_enu       = N.ma.array(N.zeros(extshape), mask=N.zeros(extshape))
        self.psur_ext_y_enu       = self.psur_ext_x_enu.copy()
        self.psur_ext_x_edep      = self.psur_ext_x_enu.copy()
        self.psur_ext_y_edep      = self.psur_ext_x_enu.copy()
        self.psur_ext_x_edep_lsnl = self.psur_ext_x_enu.copy()
        self.psur_ext_y_edep_lsnl = self.psur_ext_x_enu.copy()

    def set_eres(self, fcn):
        self.eres_edep      = fcn(self.psur_ext_x_edep)
        self.eres_edep_lsnl = fcn(self.psur_ext_x_edep_lsnl)

        self.rel_psur_ext_y_edep = self.psur_ext_y_edep/self.eres_edep
        self.rel_psur_ext_y_edep_lsnl = self.psur_ext_y_edep_lsnl/self.eres_edep_lsnl

#
# Plots and tests
#
def main(opts):
    global savefig

    if opts.output and opts.output.endswith('.pdf'):
        pdfpages = PdfPages(opts.output)
        pdfpagesfilename=opts.output
        savefig_old=savefig
        pdf=pdfpages.__enter__()
        def savefig(*args, **kwargs):
            close = kwargs.pop('close', False)
            if opts.individual and args and args[0]:
                savefig_old(*args, **kwargs)
            pdf.savefig()
            if close:
                plt.close()
    else:
        pdf = None
        pdfpagesfilename = ''
        pdfpages = None

    cfg = NestedDict(
        bundle = dict(
            name='energy_nonlinearity_birks_cherenkov',
            version='v01',
            nidx=[ ('r', 'reference', ['R1', 'R2']) ],
            major=[],
            ),
        stopping_power='data/data_juno/energy_model/2019_birks_cherenkov_v01/stoppingpower.txt',
        annihilation_electrons=dict(
            file='data/data_juno/energy_model/2019_birks_cherenkov_v01/hgamma2e.root',
            histogram='hgamma2e_1KeV',
                            scale=1.0/50000 # events simulated
            ),
        pars = uncertaindict(
            [
                ('birks.Kb0',               (1.0, 'fixed')),
                ('birks.Kb1',           (15.2e-3, 0.1776)),
                # ('birks.Kb2',           (0.0, 'fixed')),
                ("cherenkov.E_0",         (0.165, 'fixed')),
                ("cherenkov.p0",  ( -7.26624e+00, 'fixed')),
                ("cherenkov.p1",   ( 1.72463e+01, 'fixed')),
                ("cherenkov.p2",  ( -2.18044e+01, 'fixed')),
                ("cherenkov.p3",   ( 1.44731e+01, 'fixed')),
                ("cherenkov.p4",   ( 3.22121e-02, 'fixed')),
                ("Npescint",            (1341.38, 0.0059)),
                ("kC",                      (0.5, 0.4737)),
                ("normalizationEnergy",   (11.999, 'fixed'))
             ],
            mode='relative'
            ),
        integration_order = 2,
        correlations_pars = [ 'birks.Kb1', 'Npescint', 'kC' ],
        correlations = [ 1.0,   0.94, -0.97,
                         0.94,  1.0,  -0.985,
                        -0.97, -0.985, 1.0   ],
        fill_matrix=True,
        labels = dict(
            normalizationEnergy = 'Pessimistic'
            ),
        )

    ns = env.globalns('energy')
    quench = execute_bundle(cfg, namespace=ns)
    print()
    normE = ns['normalizationEnergy'].value()

    #
    # Input bins
    #
    evis_edges_full_input = N.arange(0.0, 15.0+1.e-6, 0.025)
    evis_edges_full_hist = C.Histogram(evis_edges_full_input, labels='Evis bin edges')
    evis_edges_full_hist >> quench.context.inputs.evis_edges_hist['00']

    #
    # Python energy model interpolation function
    #
    from scipy.interpolate import interp1d
    lsnl_x = quench.histoffset.histedges.points_truncated.data()
    lsnl_y = quench.positron_model_relative.single().data()
    lsnl_fcn = interp1d(lsnl_x, lsnl_y, kind='quadratic')

    #
    # Energy resolution
    #
    def eres_sigma_rel(edep):
        return 0.03/edep**0.5

    def eres_sigma_abs(edep):
        return 0.03*edep**0.5

    #
    # Energy offset
    #
    from physlib import pc
    edep_offset = pc.DeltaNP - pc.ElectronMass

    #
    # Oscprob
    #
    baselinename='L'
    ns = env.ns("oscprob")
    import gna.parameters.oscillation
    gna.parameters.oscillation.reqparameters(ns)
    ns.defparameter(baselinename, central=52.0, fixed=True, label='Baseline, km')

    #
    # Define energy range
    #
    enu_input = N.arange(1.8, 12.0, 0.001)
    edep_input = enu_input - edep_offset
    edep_lsnl = edep_input * lsnl_fcn(edep_input)
    dmrange = N.linspace(2.4e-3, 2.6e-3, 21)
    dmmid_idx = int((dmrange.size-1)//2)
    dshape   = (dmrange.size, enu_input.size)
    extshape = (dmrange.size, 100)

    # Initialize oscillation variables
    enu = C.Points(enu_input, labels='Neutrino energy, MeV')
    component_names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
    with ns:
        R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), component_names, ns=ns)

        labels=['Oscillation probability|%s'%s for s in ('component 12', 'component 13', 'component 23', 'full', 'probsum')]
        oscprob = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae(), baselinename, labels=labels)

    enu >> oscprob.full_osc_prob.Enu
    enu >> (oscprob.comp12.Enu, oscprob.comp13.Enu, oscprob.comp23.Enu)

    unity = C.FillLike(1, labels='Unity')
    enu >> unity.fill.inputs[0]
    with ns:
        op_sum = C.WeightedSum(component_names, [unity.fill.outputs[0], oscprob.comp12.comp12, oscprob.comp13.comp13, oscprob.comp23.comp23], labels='Oscillation probability sum')

    oscprob.printtransformations()
    env.globalns.printparameters(labels=True)

    #
    # Positron non-linearity
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Evis/Edep', title='Positron energy nonlineairty')
    ax.minorticks_on(); ax.grid()
    quench.positron_model_relative.single().plot_vs(quench.histoffset.histedges.points_truncated, label='definition range')
    quench.positron_model_relative_full.plot_vs(quench.histoffset.histedges.points, '--', linewidth=1., label='full range', zorder=0.5)
    ax.vlines(normE, 0.0, 1.0, linestyle=':')
    ax.legend(loc='lower right')
    ax.set_ylim(0.8, 1.05)
    savefig(opts.output, suffix='_total_relative', close=True)

    #
    # Energy resolution
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel=r'$\sigma/E$', title='Energy resolution')
    ax.minorticks_on(); ax.grid()
    ax.plot(edep_input, eres_sigma_rel(edep_input), '-')
    savefig(opts.output, suffix='_eres_rel', close=True)

    #
    # Energy resolution
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel= 'Edep, MeV', ylabel= r'$\sigma$', title='Energy resolution')
    ax.minorticks_on(); ax.grid()
    ax.plot(edep_input, eres_sigma_abs(edep_input), '-')
    savefig(opts.output, suffix='_eres_abs', close=True)

    # Calculation function
    data_no = Data(dshape, extshape)
    data_io = Data(dshape, extshape)
    def calc(data, i):
        psuri = data.psur[i] = op_sum.single().data()
        psur_minima, = argrelmin(psuri)
        psur_maxima, = argrelmax(psuri)

        def build_extrema(x, target_x, target_y):
            data_min_x = (x[psur_minima][:-1] + x[psur_minima][1:])*0.5
            data_min_y = (x[psur_minima][1:] - x[psur_minima][:-1])

            data_max_x = (x[psur_maxima][:-1] + x[psur_maxima][1:])*0.5
            data_max_y = (x[psur_maxima][1:] - x[psur_maxima][:-1])

            data_ext_x, data_ext_y = merge_extrema(data_min_x, data_max_x, data_min_y, data_max_y)

            target_x[:data_ext_x.size] = data_ext_x
            target_x.mask[:data_ext_x.size] = False
            target_x.mask[data_ext_x.size:] = True

            target_y[:data_ext_y.size] = data_ext_y
            target_y.mask[:data_ext_y.size] = False
            target_y.mask[data_ext_y.size:] = True

        build_extrema(enu_input, data.psur_ext_x_enu[i], data.psur_ext_y_enu[i])
        build_extrema(edep_input, data.psur_ext_x_edep[i], data.psur_ext_y_edep[i])
        build_extrema(edep_lsnl, data.psur_ext_x_edep_lsnl[i], data.psur_ext_y_edep_lsnl[i])

    def calcall(nmo):
        ns = env.globalns('oscprob')
        dmpar = ns['DeltaMSqEE']
        alpha = ns['Alpha']
        prev = alpha.value()
        alpha.set(nmo)
        dmpar.push()
        data = nmo=='normal' and data_no or data_io
        for i, dm in enumerate(dmrange):
            dmpar.set(dm)
            calc(data, i)
        dmpar.pop()
        alpha.set(prev)
    calcall('normal')
    calcall('inverted')

    #
    # Survival probability vs Enu
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Enu, MeV', ylabel='Psur', title='Survival probability')
    ax.minorticks_on(); ax.grid()
    ax.plot(enu.single().data(), data_no.psur[dmmid_idx], label=r'full NO')
    ax.plot(enu.single().data(), data_io.psur[dmmid_idx], label=r'full IO')
    # ax.plot(enu_input[data_no.psur_minima], data_no.psur[data_no.psur_minima], 'o', markerfacecolor='none', label='minima')
    # ax.plot(enu_input[data_no.psur_maxima], data_no.psur[data_no.psur_maxima], 'o', markerfacecolor='none', label='maxima')
    savefig(opts.output, suffix='_psur_enu', close=True)

    #
    # Survival probability vs Edep
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Psur', title='Survival probability')
    ax.minorticks_on(); ax.grid()
    ax.plot(edep_input, data_no.psur[dmmid_idx], label='true NO')
    ax.plot(edep_lsnl,  data_no.psur[dmmid_idx], label=r'with LSNL NO')
    ax.plot(edep_lsnl,  data_io.psur[dmmid_idx], label=r'with LSNL IO')
    ax.legend()
    savefig(opts.output, suffix='_psur_edep')

    #
    # Distance between nearest peaks vs Enu
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Enu, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    ax.plot(data_no.psur_ext_x_enu[dmmid_idx], data_no.psur_ext_y_enu[dmmid_idx], 'o-', markerfacecolor='none')
    savefig(opts.output, suffix='_dist_enu', close=True)

    #
    # Distance between nearest peaks vs Edep, single
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    ax.plot( data_no.psur_ext_x_edep[dmmid_idx], data_no.psur_ext_y_edep[dmmid_idx], '-', markerfacecolor='none', label='true' )
    ax.plot( data_no.psur_ext_x_edep_lsnl[dmmid_idx], data_no.psur_ext_y_edep_lsnl[dmmid_idx], '-', markerfacecolor='none', label='with LSNL' )
    ax.plot( edep_input, eres_sigma_abs(edep_input), '-', markerfacecolor='none', label=r'$\sigma$' )
    ax.legend(loc='upper left')
    savefig(opts.output, suffix='_dist')

    #
    # Distance between nearest peaks vs Edep
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    stride=10
    ax.plot( data_no.psur_ext_x_edep.T[:,::stride], data_no.psur_ext_y_edep.T[:,::stride], '-',  color='blue', markerfacecolor='none', alpha=0.5 )
    ax.plot( data_io.psur_ext_x_edep.T[:,::stride], data_io.psur_ext_y_edep.T[:,::stride], '--',  color='red', markerfacecolor='none', alpha=0.5 )
    # ax.plot( data_no.psur_ext_x_edep_lsnl.T[:,::stride], data_no.psur_ext_y_edep_lsnl.T[:,::stride], '--', color='green', markerfacecolor='none', alpha=0.5 )
    # ax.plot( edep_input, eres_sigma_abs(edep_input), '-', markerfacecolor='none', label=r'$\sigma$' )
    ax.legend(loc='upper left')
    savefig(opts.output, suffix='_dist')

    data_no.set_eres(eres_sigma_abs)
    data_io.set_eres(eres_sigma_abs)
    #
    # Resolution ability, single
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel=r'Dist/$\sigma$', title='Resolution ability')
    ax.minorticks_on(); ax.grid()
    ax.plot(data_no.psur_ext_x_edep[dmmid_idx], data_no.rel_psur_ext_y_edep[dmmid_idx], '-',  color='blue', markerfacecolor='none')
    ax.plot(data_io.psur_ext_x_edep[dmmid_idx], data_io.rel_psur_ext_y_edep[dmmid_idx], '--', color='red', markerfacecolor='none')
    ax.legend(loc='upper left')
    savefig(opts.output, suffix='_ability')

    #
    # Resolution ability
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel=r'Dist/$\sigma$', title='Resolution ability')
    ax.minorticks_on(); ax.grid()
    stride=10
    ax.plot(data_no.psur_ext_x_edep.T[:,::stride], data_no.rel_psur_ext_y_edep.T[:,::stride], '-',  color='blue', markerfacecolor='none')
    ax.plot(data_io.psur_ext_x_edep.T[:,::stride], data_io.rel_psur_ext_y_edep.T[:,::stride], '--', color='red', markerfacecolor='none')
    ax.legend(loc='upper left')
    savefig(opts.output, suffix='_ability')

    #
    # Resolution ability
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel=r'Dist/$\sigma$', title='Resolution ability LSNL')
    ax.minorticks_on(); ax.grid()
    stride=10
    ax.plot(data_no.psur_ext_x_edep_lsnl.T[:,::stride], data_no.rel_psur_ext_y_edep_lsnl.T[:,::stride], '-',  color='blue', markerfacecolor='none')
    ax.plot(data_io.psur_ext_x_edep_lsnl.T[:,::stride], data_io.rel_psur_ext_y_edep_lsnl.T[:,::stride], '--', color='red', markerfacecolor='none')
    ax.legend(loc='upper left')
    savefig(opts.output, suffix='_ability')

    # ax.set_xlim(3, 4)
    # ax.set_ylim(5, 8)
    # savefig(opts.output, suffix='_ability_zoom')

    # fig = P.figure()
    # ax = P.subplot( 111 )
    # ax.minorticks_on()
    # ax.grid()
    # ax.set_xlabel( 'Edep, MeV' )
    # ax.set_ylabel( r'Dist/$\sigma$' )
    # ax.set_title( 'Resolution ability difference (quenching-true)' )

    # y2fcn = interp1d(x2, y2)
    # y2_on_x1 = y2fcn(x1)
    # diff = y2_on_x1 - y1
    # from scipy.signal import savgol_filter
    # diff = savgol_filter(diff, 21, 3)

    # ax.plot(x1, diff)

    # savefig(opts.output, suffix='_ability_diff')

    # if pdfpages:
        # pdfpages.__exit__(None,None,None)
        # print('Write output figure to', pdfpagesfilename)

    # savegraph(quench.histoffset.histedges.points_truncated, opts.graph, namespace=ns)

    if opts.show:
        P.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-i', '--individual', help='Save individual output files', action='store_true')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('-s', '--show', action='store_true', help='Show the plots')
    parser.add_argument('-m', '--mapping', action='store_true', help='Do mapping plot')

    main( parser.parse_args() )
