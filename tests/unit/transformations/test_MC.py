#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig, add_colorbar, plot_hist, plot_hist_errorbar
import pytest
from gna.unittest import allure_attach_file, savegraph
from gna import constructors as C
from gna.bindings import common
import os
from argparse import Namespace

class MCTestData(object):
    mcdata = None
    corrmat = None
    figures = tuple()
    correlation  = 0.95
    syst_unc_rel = 2
    nsigma = 4
    def __init__(self, data, mctype, **info):
        self.info     = Namespace(**info)
        self.info.index = str(self.info.index)

        self.mctype   = mctype
        self.data     = data
        self.shape2   = (data.size, data.size)
        self.err_stat2 = data.copy()
        self.err_stat = self.err_stat2**0.5

        self.edges = np.arange(self.data.size+1, dtype='d')
        self.hist = C.Histogram(self.edges, self.data)

        if mctype=='CovarianceToyMC':
            self.prepare_corrmatrix()
            self.prepare_covmatrix_syst()
            self.prepare_covmatrix_full()

        self.prepare_inputs()

    def prepare_inputs(self):
        if self.mctype=='NormalToyMC':
            self.input_err = C.Points(self.err_stat)
            self.inputs = (self.hist, self.input_err)
        elif self.mctype=='CovarianceToyMC':
            self.inputs = (self.hist, self.input_L)
        else:
            self.inputs = (self.hist,)

    def prepare_corrmatrix(self):
        self.corrmat = np.full(self.shape2, self.correlation, dtype='d')
        part = self.data.size//2
        self.corrmat[:part, part:] = -self.corrmat[:part, part:]
        self.corrmat[part:, :part] = -self.corrmat[part:, :part]

        np.fill_diagonal(self.corrmat, 1.0)

    def prepare_covmatrix_syst(self):
        self.err_syst     = self.syst_unc_rel*self.data
        self.err_syst_sqr = np.diag(self.err_syst**0.5)
        self.covmat_syst = np.dot(np.dot(self.err_syst_sqr.T, self.corrmat), self.err_syst_sqr)

    def prepare_covmatrix_full(self):
        self.covmat_full = np.diag(self.err_stat2) + self.covmat_syst
        self.covmat_L    = np.linalg.cholesky(self.covmat_full)
        self.covmat_L_inv= np.linalg.inv(self.covmat_L)

        self.input_L = C.Points(self.covmat_L)

    def set_mc(self, mcoutput):
        self.mcoutput = mcoutput
        self.mcdata = mcoutput.data()
        self.mcdiff = self.mcdata - self.data

        if self.corrmat is None:
            self.mcdiff_norm = self.mcdiff/self.err_stat
        else:
            self.mcdiff_norm = np.dot(self.covmat_L_inv, self.mcdiff)

    def plot(self, tmp_path):
        assert self.mcdata is not None
        self.tmp_path = tmp_path

        self.plot_hist()
        self.plot_mats()

    def figure(self, *args, **kwargs):
        fig = plt.figure(*args, **kwargs)
        self.figures += fig,

        return fig

    def close(self):
        map(plt.close, self.figures)

    def savefig(self, *args):
        path = '_'.join((self.tmp_path,)+tuple(args))+'.png'
        savefig(path, dpi=300)
        allure_attach_file(path)

    def plot_hist(self):
        fig = self.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( '' )
        ax.set_ylabel( '' )
        ax.set_title('Check {index}, input {}, scale {scale}'.format(self.mctype, **self.info.__dict__))

        self.hist.plot_hist(color='black', label='input')
        self.mcoutput.plot_errorbar(yerr='stat', linestyle='--', label='output')

        ax.legend()

        self.savefig('hist', self.info.index)

        fig = self.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( '' )
        ax.set_ylabel( '' )
        ax.set_title('Check diff {index}, input {}, scale {scale}'.format(self.mctype, **self.info.__dict__))

        plot_hist_errorbar(self.edges, self.mcdiff, self.err_stat, label='raw difference')
        plot_hist_errorbar(self.edges, self.mcdiff_norm, 1.0, label='normalized uncorrelated')

        ax.legend()

        self.savefig('diff', self.info.index)

        ax.set_ylim(-4,4)
        self.savefig('diff_zoom', self.info.index)

    def matshow(self, mat, title, suffix):
        fig = self.figure()
        ax = plt.subplot( 111 )
        c = plt.matshow(mat, fignum=False)
        add_colorbar(c)
        ax = plt.gca()
        ax.set_title(title)

        self.savefig(suffix, self.info.index)

    def plot_mats(self):
        if self.corrmat is None:
            return

        self.matshow(self.corrmat,     'Correlation matrix', 'corrmat')
        self.matshow(self.covmat_syst, 'Covariance matrix (syst)', 'covmat_syst')
        self.matshow(self.covmat_full, 'Covariance matrix (full)', 'covmat_full')
        self.matshow(self.covmat_L,    'Covariance matrix decomposed: L', 'covmat_L')

    def check_stats(self):
        if self.mctype=='Snapshot':
            assert (self.mcdiff==0.0).all()
        else:
            if self.mctype!='PoissonToyMC':
                assert (self.mcdiff!=0.0).all()

            mcdiff_abs = np.fabs(self.mcdiff_norm)
            assert (mcdiff_abs<self.nsigma).all()

            sum  = self.mcdiff_norm.sum()
            sum_abs = np.fabs(sum)
            assert sum_abs < self.nsigma * self.data.size**0.5

            chi2 = (self.mcdiff_norm**2).sum()
            chi2_diff = chi2 - self.data.size
            assert chi2_diff < self.nsigma*(2.0*self.data.size)**0.5

@pytest.mark.parametrize('scale', [0.1, 100.0, 10000.0])
@pytest.mark.parametrize('mc', ['Snapshot', 'PoissonToyMC', 'NormalStatsToyMC', 'NormalToyMC', 'CovarianceToyMC'])
def test_mc(mc, scale, tmp_path):
    R.GNA.Random.seed( 3 ) # With 4 sigma acceptance level failures do happen. Fix the random seed to guarantee the correct behaviour.
    size = 20
    data1 = np.ones(size, dtype='d')*scale
    data2 = (1.0+np.arange(size, dtype='d'))*scale
    data3 = (size - np.arange(size, dtype='d'))*scale

    mcdata_v = tuple(MCTestData(data, mc, index=i, scale=scale) for i, data in enumerate((data1, data2, data3)))

    MCclass = getattr(C, mc)
    mcobject = MCclass()

    for mcdata in mcdata_v:
        mcobject.add_inputs(*mcdata.inputs)

    mcobject.printtransformations()

    map(lambda (o, out): MCTestData.set_mc(o, out), zip(mcdata_v, mcobject.transformations[0].outputs.values()))

    tmp_path = os.path.join(str(tmp_path), '_'.join((mc, str(int(scale)))) )
    map(lambda o: MCTestData.plot(o, tmp_path), mcdata_v)

    map(MCTestData.close, mcdata_v)
    map(MCTestData.check_stats, mcdata_v)

