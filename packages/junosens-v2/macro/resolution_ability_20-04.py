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
from scipy.interpolate import interp1d

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

class DataE(object):
    def __init__(self, dshape, e, eres_fcn):
        self.dshape=dshape
        self.extshape=(dshape[0], 100)
        self.eres_fcn=eres_fcn

        self.e      = e
        self.diff_x = N.ma.array(N.zeros(self.extshape), mask = N.zeros(self.extshape))
        self.diff   = self.diff_x.copy()
        self.diff_interp = [None]*dshape[0]
        self.eres   = self.diff_x.copy()
        self.psur_e = self.diff_x.copy()
        self.psur   = self.diff_x.copy()

    def build_datum(self, ext_idx):
        data_x = (self.e[ext_idx][:-1] + self.e[ext_idx][1:])*0.5
        data_y = (self.e[ext_idx][1:] - self.e[ext_idx][:-1])
        return data_x, data_y

    def build_data(self, i, psur, minima_idx, maxima_idx):
        diff_x, diff = self.build_datum(maxima_idx)
        target_x, target_y = self.diff_x[i], self.diff[i]

        size = diff_x.size
        target_x[:size] = diff_x
        target_y[:size] = diff
        target_x.mask[:size] = False
        target_x.mask[size:] = True
        target_y.mask = target_x.mask

        size+=1
        self.psur_e[i][:size] = self.e[maxima_idx]
        self.psur_e[i].mask[:size] = False
        self.psur_e[i].mask[size:] = True
        self.psur[i][:size] = psur[maxima_idx]
        self.psur[i].mask[:size] = False
        self.psur[i].mask[size:] = True

        self.eres[i] = self.eres_fcn(target_x)

        interp = interp1d(diff_x, diff, kind='quadratic', bounds_error=False)
        self.diff_interp[i] = interp

class DataNMO(object):
    def __init__(self, dshape, enu, edep, edep_lsnl, eres_fcn, nmo):
        self.psur           = N.zeros(dshape)
        self.data_enu       = DataE(dshape, enu, eres_fcn)
        self.data_edep      = DataE(dshape, edep, eres_fcn)
        self.data_edep_lsnl = DataE(dshape, edep_lsnl, eres_fcn)
        self.nmo            = nmo

        self.data           = (self.data_enu, self.data_edep, self.data_edep_lsnl)

        self.diffs = Namespace()

    def build_data(self, i, psuri):
        self.psur[i] = psuri
        minima_idx, = argrelmin(psuri)
        maxima_idx, = argrelmax(psuri)

        self.data_enu.build_data(i, psuri, minima_idx, maxima_idx)
        self.data_edep.build_data(i, psuri, minima_idx, maxima_idx)
        self.data_edep_lsnl.build_data(i, psuri, minima_idx, maxima_idx)

class Data(object):
    fcn = None
    diffs = None
    def __init__(self, enu, lsnl_fcn, eres_fcn):
        self.enu     = enu
        self.dmrange = N.linspace(2.4e-3, 2.6e-3, 21)
        self.dmmid_idx = int((self.dmrange.size-1)//2)
        self.dshape  = (self.dmrange.size, enu.size)
        self.lsnl_fcn = lsnl_fcn
        self.eres_fcn = eres_fcn

        from physlib import pc
        edep_offset = pc.DeltaNP - pc.ElectronMass
        self.edep = self.enu - edep_offset
        self.edep_lsnl = self.edep * lsnl_fcn(self.edep)

        self.data_no = DataNMO(self.dshape, self.enu, self.edep, self.edep_lsnl, eres_fcn=eres_fcn, nmo='normal')
        self.data_io = DataNMO(self.dshape, self.enu, self.edep, self.edep_lsnl, eres_fcn=eres_fcn, nmo='inverted')

    def set_psur_fcn(self, fcn):
        self.psur_fcn = fcn

    def set_dm_par(self, dmpar):
        self.dm_par = dmpar

    def set_nmo_par(self, nmopar):
        self.nmo_par = nmopar

    def build(self):
        self.nmo_par.push()
        self.dm_par.push()
        for data in (self.data_no, self.data_io):
            self.nmo_par.set(data.nmo)
            for i, dm in enumerate(self.dmrange):
                self.dm_par.set(dm)
                data.build_data(i, self.psur_fcn())
        self.dm_par.pop()
        self.nmo_par.pop()
        self.build_diffs()

    def build_diffs(self):
        self.e = self.edep
        self.mesh_dm, self.mesh_e = N.meshgrid(self.dmrange, self.e, indexing='ij')

        self.diffs = Namespace()
        self.diffs_rel = Namespace()
        self.eres = self.eres_fcn(self.e)
        for energy in ('enu', 'edep', 'edep_lsnl'):
            diffs     = N.zeros(self.dshape)
            diffs_rel = N.zeros(self.dshape)
            diff_no_all = self.data_no.__dict__['data_'+energy].diff_interp
            diff_io_all = self.data_io.__dict__['data_'+energy].diff_interp
            for i, (no, io) in enumerate(zip(diff_no_all, diff_io_all)):
                diff_no = no(self.e)
                diff_io = io(self.e)
                diffs[i] = diff_io - diff_no
                diffs_rel[i] = diffs[i]/self.eres

            self.diffs.__dict__[energy]     = N.ma.array(diffs, mask=N.isnan(diffs))
            self.diffs_rel.__dict__[energy] = N.ma.array(diffs_rel, mask=N.isnan(diffs_rel))

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
                P.close()
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
                ("normalizationEnergy",   (11.9999999, 'fixed'))
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
    evis_edges_full_input = N.arange(0.0, 15.0+1.e-6, 0.001)
    evis_edges_full_hist = C.Histogram(evis_edges_full_input, labels='Evis bin edges')
    evis_edges_full_hist >> quench.context.inputs.evis_edges_hist['00']

    #
    # Python energy model interpolation function
    #
    lsnl_x = quench.histoffset.histedges.points_truncated.data()
    lsnl_y = quench.positron_model_relative.single().data()
    lsnl_fcn = interp1d(lsnl_x, lsnl_y, kind='quadratic', bounds_error=False, fill_value='extrapolate')

    #
    # Energy resolution
    #
    def eres_sigma_rel(edep):
        return 0.03/edep**0.5

    def eres_sigma_abs(edep):
        return 0.03*edep**0.5

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
    data = Data(N.arange(1.8, 15.0, 0.001), lsnl_fcn=lsnl_fcn, eres_fcn=eres_sigma_abs)

    # Initialize oscillation variables
    enu = C.Points(data.enu, labels='Neutrino energy, MeV')
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

    ns = env.globalns('oscprob')
    data.set_dm_par(ns['DeltaMSqEE'])
    data.set_nmo_par(ns['Alpha'])
    data.set_psur_fcn(op_sum.single().data)
    data.build()

    #
    # Plotting
    #
    xmax = 12.0

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
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_total_relative', close=not opts.show_all)

    #
    # Positron non-linearity derivative
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='dEvis/dEdep', title='Positron energy nonlineairty derivative')
    ax.minorticks_on(); ax.grid()
    e = quench.histoffset.histedges.points_truncated.single().data()
    f = quench.positron_model_relative.single().data()*e
    ec = (e[1:] + e[:-1])*0.5
    df = (f[1:] - f[:-1])
    dedf = (e[1:] - e[:-1])/df
    ax.plot(ec, dedf)
    ax.legend(loc='lower right')
    ax.set_ylim(0.975, 1.01)
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_total_derivative', close=not opts.show_all)

    #
    # Positron non-linearity effect
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Evis/Edep', title='Positron energy nonlineairty')
    ax.minorticks_on(); ax.grid()

    es = N.arange(1.0, 3.1, 0.5)
    esmod = es*lsnl_fcn(es)
    esmod_shifted = esmod*(es[-1]/esmod[-1])
    ax.vlines(es, 0.0, 1.0, linestyle='--', linewidth=2, alpha=0.5, color='green', label='Edep')
    ax.vlines(esmod, 0.0, 1.0, linestyle='-', color='red', label='Edep quenched')
    ax.legend()
    savefig(opts.output, suffix='_quenching_effect_0')

    ax.vlines(esmod_shifted, 0.0, 1.0, linestyle=':', color='blue', label='Edep quenched, scaled')
    ax.legend()
    savefig(opts.output, suffix='_quenching_effect_1', close=not opts.show_all)

    #
    # Energy resolution
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel=r'$\sigma/E$', title='Energy resolution')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.edep, eres_sigma_rel(data.edep), '-')
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_eres_rel', close=not opts.show_all)

    #
    # Energy resolution
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel= 'Edep, MeV', ylabel= r'$\sigma$', title='Energy resolution')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.edep, eres_sigma_abs(data.edep), '-')
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_eres_abs', close=not opts.show_all)

    #
    # Survival probability vs Enu
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Enu, MeV', ylabel='Psur', title='Survival probability')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.enu, data.data_no.psur[data.dmmid_idx], label=r'full NO')
    ax.plot(data.enu, data.data_io.psur[data.dmmid_idx], label=r'full IO')
    ax.plot(data.data_no.data_enu.psur_e[data.dmmid_idx], data.data_no.data_enu.psur[data.dmmid_idx], '^', markerfacecolor='none')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_psur_enu')

    ax.set_xlim(2.0, 4.5)
    savefig(opts.output, suffix='_psur_enu_zoom', close=not opts.show_all)

    #
    # Survival probability vs Edep
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Psur', title='Survival probability')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.edep, data.data_no.psur[data.dmmid_idx], label=r'full NO')
    ax.plot(data.edep, data.data_io.psur[data.dmmid_idx], label=r'full IO')
    ax.plot(data.data_no.data_edep.psur_e[data.dmmid_idx], data.data_no.data_edep.psur[data.dmmid_idx], '^', markerfacecolor='none')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_psur_edep')

    ax.set_xlim(1.2, 3.7)
    savefig(opts.output, suffix='_psur_edep_zoom', close=not opts.show_all)

    #
    # Survival probability vs Edep_lsnl
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep quenched, MeV', ylabel='Psur', title='Survival probability')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.edep_lsnl, data.data_no.psur[data.dmmid_idx], label=r'full NO')
    ax.plot(data.edep_lsnl, data.data_io.psur[data.dmmid_idx], label=r'full IO')
    ax.plot(data.data_no.data_edep_lsnl.psur_e[data.dmmid_idx], data.data_no.data_edep_lsnl.psur[data.dmmid_idx], '^', markerfacecolor='none')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_psur_edep_lsnl')

    ax.set_xlim(1.2, 3.7)
    savefig(opts.output, suffix='_psur_edep_lsnl_zoom', close=not opts.show_all)

    #
    # Distance between nearest peaks vs Enu, single
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Enu, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.data_no.data_enu.diff_x[data.dmmid_idx], data.data_no.data_enu.diff[data.dmmid_idx], label=r'NO')
    ax.plot(data.data_io.data_enu.diff_x[data.dmmid_idx], data.data_io.data_enu.diff[data.dmmid_idx], label=r'IO')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    savefig(opts.output, suffix='_dist_enu')

    ax.set_xlim(2.0, 5.0)
    ax.set_ylim(top=0.5)
    savefig(opts.output, suffix='_dist_enu_zoom', close=not opts.show_all)

    #
    # Distance between nearest peaks vs Edep, single
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.data_no.data_edep.diff_x[data.dmmid_idx], data.data_no.data_edep.diff[data.dmmid_idx], label=r'NO')
    ax.plot(data.data_io.data_edep.diff_x[data.dmmid_idx], data.data_io.data_edep.diff[data.dmmid_idx], label=r'IO')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    savefig(opts.output, suffix='_dist_edep')

    ax.set_xlim(1.2, 4.2)
    ax.set_ylim(top=0.5)
    savefig(opts.output, suffix='_dist_edep_zoom', close=not opts.show_all)

    #
    # Distance between nearest peaks vs Edep, single
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep quenched, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.data_no.data_edep_lsnl.diff_x[data.dmmid_idx], data.data_no.data_edep_lsnl.diff[data.dmmid_idx], label=r'NO')
    ax.plot(data.data_io.data_edep_lsnl.diff_x[data.dmmid_idx], data.data_io.data_edep_lsnl.diff[data.dmmid_idx], label=r'IO')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    savefig(opts.output, suffix='_dist_edep_lsnl')

    ax.set_xlim(1.2, 4.2)
    ax.set_ylim(top=0.5)
    savefig(opts.output, suffix='_dist_edep_lsnl_zoom')

    poly = N.polynomial.polynomial.Polynomial([0, 1, 0])
    x = data.data_no.data_edep_lsnl.diff_x[data.dmmid_idx]
    pf = poly.fit(x, data.data_no.data_edep_lsnl.diff[data.dmmid_idx], 2)
    print(pf)
    ax.plot(x, pf(x), label=r'NO fit')
    ax.legend()
    savefig(opts.output, suffix='_dist_edep_lsnl_fit', close=not opts.show_all)

    #
    # Distance between nearest peaks vs Edep, multiple
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep quenched, MeV', ylabel='Dist, MeV', title='Nearest peaks distance')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.data_no.data_edep_lsnl.diff_x[data.dmmid_idx], data.data_no.data_edep_lsnl.diff[data.dmmid_idx], label=r'NO')
    ax.plot(data.data_io.data_edep_lsnl.diff_x[data.dmmid_idx], data.data_io.data_edep_lsnl.diff[data.dmmid_idx], '--', label=r'IO')
    for idx in (0, 5, 15, 20):
        ax.plot(data.data_io.data_edep_lsnl.diff_x[idx], data.data_io.data_edep_lsnl.diff[idx], '--')
    ax.legend()
    ax.set_xlim(0.0, xmax)
    savefig(opts.output, suffix='_dist_edep_lsnl_multi', close=not opts.show_all)

    #
    # Distance between nearest peaks difference
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='Dist(IO) - Dist(NO), MeV', title='Nearest peaks distance diff: IO-NO')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.e, data.diffs.edep[data.dmmid_idx], '-', markerfacecolor='none', label='Edep')
    ax.plot(data.e, data.diffs.edep_lsnl[data.dmmid_idx], '-', markerfacecolor='none', label='Edep quenched')
    ax.plot(data.e, data.diffs.enu[data.dmmid_idx], '-', markerfacecolor='none', label='Enu')
    ax.legend()
    savefig(opts.output, suffix='_dist_diff')

    ax.plot(data.e, data.eres, '-', markerfacecolor='none', label='Resolution $\\sigma$')
    ax.legend()
    savefig(opts.output, suffix='_dist_diff_1')

    #
    # Distance between nearest peaks difference relative to sigma
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='(Dist(IO) - Dist(NO))/$\\sigma$', title='Nearest peaks distance diff: IO-NO')
    ax.minorticks_on(); ax.grid()
    ax.plot(data.e, data.diffs_rel.edep[data.dmmid_idx], '-', markerfacecolor='none', label='Edep')
    ax.plot(data.e, data.diffs_rel.edep_lsnl[data.dmmid_idx], '-', markerfacecolor='none', label='Edep quenched')

    i_edep=N.argmax(data.diffs_rel.edep[data.dmmid_idx])
    i_edep_lsnl=N.argmax(data.diffs_rel.edep_lsnl[data.dmmid_idx])
    ediff = data.e[i_edep] - data.e[i_edep_lsnl]
    ax.axvline(data.e[i_edep], linestyle='dashed', label='Max location diff: %.2f MeV'%(ediff))
    ax.axvline(data.e[i_edep_lsnl], linestyle='dashed')

    ax.legend()
    savefig(opts.output, suffix='_dist_diff_rel')

    #
    # Distance between nearest peaks difference relative to sigma
    #
    fig = P.figure()
    ax = P.subplot(111, xlabel='Edep, MeV', ylabel='(Dist(IO) - Dist(NO))/$\\sigma$', title='Nearest peaks distance diff: IO-NO')
    ax.minorticks_on(); ax.grid()
    ledep   = ax.plot(data.e, data.diffs_rel.edep[data.dmmid_idx],      '--', markerfacecolor='none', label='Edep')[0]
    lquench = ax.plot(data.e, data.diffs_rel.edep_lsnl[data.dmmid_idx], '-',  color=ledep.get_color(), markerfacecolor='none', label='Edep quenched')[0]

    kwargs=dict(alpha=0.8, linewidth=1.5, markerfacecolor='none')
    for idx in (0, 5, 15, 20):
        l = ax.plot(data.e, data.diffs_rel.edep[idx], '--', **kwargs)[0]
        ax.plot(data.e, data.diffs_rel.edep_lsnl[idx], '-', color=l.get_color(), **kwargs)
    ax.legend()
    savefig(opts.output, suffix='_dist_diff_rel_multi', close=not opts.show_all)

    #
    # Distance between nearest peaks difference relative to sigma
    #
    fig = P.figure()
    ax = P.subplot(111, ylabel=r'$\Delta m^2_\mathrm{ee}$', xlabel='Edep quenched, MeV',
                   title='Nearest peaks distance diff: IO-NO/$\\sigma$')
    ax.minorticks_on(); ax.grid()
    formatter = ax.yaxis.get_major_formatter()
    formatter.set_useOffset(False)
    formatter.set_powerlimits((-2,2))
    formatter.useMathText=True

    c = ax.pcolormesh(data.mesh_e.T, data.mesh_dm.T, data.diffs_rel.edep_lsnl.T)
    from mpl_tools.helpers import add_colorbar
    add_colorbar(c, rasterized=True)
    c.set_rasterized(True)
    savefig(opts.output, suffix='_dist_diff_rel_heatmap')

    if pdfpages:
        pdfpages.__exit__(None,None,None)
        print('Write output figure to', pdfpagesfilename)

    # savegraph(quench.histoffset.histedges.points_truncated, opts.graph, namespace=ns)

    if opts.show or opts.show_all:
        P.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='Output file for images')
    parser.add_argument('-i', '--individual', help='Save individual output files', action='store_true')
    parser.add_argument('-g', '--graph', help='Output file for graph')
    parser.add_argument('-s', '--show', action='store_true', help='Show the plots')
    parser.add_argument('-m', '--mapping', action='store_true', help='Do mapping plot')
    parser.add_argument('-S', '--show-all', action='store_true', help='Show all')

    main( parser.parse_args() )
