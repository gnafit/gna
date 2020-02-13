#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna import constructors as C
from gna.env import env
from gna.unittest import *
import numpy as np
import pytest
from gna.unittest import allure_attach_file, savegraph
from matplotlib import pyplot as plt
from gna.bindings import common
from mpl_tools.helpers import savefig
import os

@pytest.mark.parametrize('diag',   [False, True])
@pytest.mark.parametrize('syst1',  [None, True])
@pytest.mark.parametrize('syst2',  [None, True])
def test_covariated_prediction(diag, syst1, syst2, tmp_path):
    if diag and (syst1 or syst2):
        return

    n = 10
    start = 10
    data = np.arange(start, start+n, dtype='d')
    stat2 = data.copy()

    fullcovmat = np.diag(stat2)

    if syst1:
        syst1 = np.ones((n,n), dtype='d')*1.5
        fullcovmat+=syst1
        cov = fullcovmat.copy()
        covbase = cov
    else:
        cov = stat2
        covbase = fullcovmat.copy()

    Data   = C.Points(data, labels='Data')
    Covmat = C.Points(cov, labels='Input covariance matrix')

    if syst2:
        syst2 = np.ones((n,n), dtype='d')*0.5
        Syst   = C.Points(syst2, labels='Input systematic matrix')
        fullcovmat+=syst2
    else:
        Syst = None

    if diag:
        Cp = C.CovariatedPredictionV(labels=['Concatenated prediction', 'Base uncertainties squared', 'Full uncertainties'])
    else:
        Cp = C.CovariatedPrediction(labels=['Concatenated prediction', 'Base covariance matrix', 'Full cov mat Cholesky decomposition'])
    Cp.append(Data)
    Cp.covariate(Covmat, Data, 1, Data, 1)
    if Syst:
        Cp.addSystematicCovMatrix(Syst)
    Cp.covbase.covbase >> Cp.cov.covbase
    Cp.finalize()

    Cp.printtransformations()

    suffix = 'covariated_prediction_{}_{}_{}'.format(diag and 'diag' or 'block', syst1 is not None and 'basediag' or 'baseblock', syst2 is not None and 'syst' or 'nosyst')

    if not diag:
        fig = plt.figure()
        ax = plt.subplot(111, xlabel='X', ylabel='Y', title='Covariance matrix base')
        ax.minorticks_on()
        ax.grid()
        Cp.covbase.covbase.plot_matshow(colorbar=True)
        path = os.path.join(str(tmp_path), suffix+'_covbase.png')
        savefig(path, dpi=300)
        allure_attach_file(path)
        plt.close()

        fig = plt.figure()
        ax = plt.subplot(111, xlabel='X', ylabel='Y', title='Covariance matrix full')
        ax.minorticks_on()
        ax.grid()
        plt.matshow(fullcovmat, fignum=False)
        path = os.path.join(str(tmp_path), suffix+'_cov.png')
        savefig(path, dpi=300)
        allure_attach_file(path)
        plt.close()

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph([Data.single(),Cp.covbase.covbase], path, verbose=False)
    allure_attach_file(path)

    data_o    = Cp.prediction.prediction.data()
    covbase_o = Cp.covbase.covbase.data()
    if diag:
        L_o      = Cp.cov.L.data()
        L_expect = stat2**0.5
    else:
        L_o       = np.tril(Cp.cov.L.data())
        L_expect = np.linalg.cholesky(fullcovmat)

    if not diag:
        fig = plt.figure()
        ax = plt.subplot(111, xlabel='X', ylabel='Y', title='L: covariance matrix decomposition')
        ax.minorticks_on()
        ax.grid()
        plt.matshow(L_o, fignum=False)
        path = os.path.join(str(tmp_path), suffix+'_L.png')
        savefig(path, dpi=300)
        allure_attach_file(path)
        plt.close()

    assert (data==data_o).all()

    if diag:
        assert (covbase_o==data).all()
    else:
        assert (covbase==covbase_o).all()

    assert np.allclose(L_o, L_expect)


