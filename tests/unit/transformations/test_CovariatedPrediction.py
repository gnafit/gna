#!/usr/bin/env python

"""Check the WeightedSum transformation"""

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
from mpl_tools.helpers import savefig, add_colorbar
import os

@pytest.mark.parametrize('syst1',  [None, True])
@pytest.mark.parametrize('syst2',  [None, True])
def test_covariated_prediction(syst1, syst2, tmp_path):
    covbase_diag=not bool(syst1)
    cov_diag=not bool(syst2) and covbase_diag

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

    Cp = C.CovariatedPrediction(labels=['Concatenated prediction', 'Base covariance matrix', 'Full cov mat Cholesky decomposition'])
    Cp.append(Data)
    Cp.covariate(Covmat, Data, 1, Data, 1)
    if Syst:
        Cp.addSystematicCovMatrix(Syst)
    Cp.covbase.covbase >> Cp.cov.covbase
    Cp.finalize()

    Cp.printtransformations()

    suffix = 'covariated_prediction_{}_{}'.format(syst1 is not None and 'baseblock' or 'basediag', syst2 is not None and 'syst' or 'nosyst')

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='X', ylabel='Y', title='Covariance matrix base')
    ax.minorticks_on()
    ax.grid()
    if covbase_diag:
        Cp.covbase.covbase.plot_hist(label='diag')
        ax.legend()
    else:
        Cp.covbase.covbase.plot_matshow(colorbar=True)
    path = os.path.join(str(tmp_path), suffix+'_covbase.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='X', ylabel='Y', title='Covariance matrix full')
    ax.minorticks_on()
    ax.grid()
    c=plt.matshow(np.ma.array(fullcovmat, mask=fullcovmat==0.0), fignum=False)
    add_colorbar(c)
    path = os.path.join(str(tmp_path), suffix+'_cov.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph([Data.single(),Cp.covbase.covbase], path, verbose=False)
    allure_attach_file(path)

    data_o    = Cp.prediction.prediction.data()
    covbase_o = Cp.covbase.covbase.data()
    if cov_diag:
        L_o      = Cp.cov.L.data()
        L_expect = stat2**0.5
    else:
        L_o       = np.tril(Cp.cov.L.data())
        L_expect = np.linalg.cholesky(fullcovmat)

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='X', ylabel='Y', title='L: covariance matrix decomposition')
    ax.minorticks_on()
    ax.grid()
    if cov_diag:
        Cp.cov.L.plot_hist(label='diag')
        ax.legend()
    else:
        c=plt.matshow(np.ma.array(L_o,mask=L_o==0.0), fignum=False)
        add_colorbar(c)
    path = os.path.join(str(tmp_path), suffix+'_L.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    assert (data==data_o).all()

    if covbase_diag:
        assert (covbase_o==data).all()
    else:
        assert (covbase==covbase_o).all()

    assert np.allclose(L_o, L_expect)

def test_covariated_prediction_blocks(tmp_path):
    ns = [1, 3, 4, 1]
    start = 10
    dataset = [np.arange(start, start+n, dtype='d') for n in ns]
    stat2set = [data.copy() for data in dataset]
    fullcovmat = [np.diag(stat2) for stat2 in stat2set]

    it = 1.0
    crosscovs=[[None]*len(ns) for i in range(len(ns))]
    Crosscovs=[[None]*len(ns) for i in range(len(ns))]
    for i in range(len(ns)):
        for j in range(i+1):
            if i==j:
                block = np.zeros((ns[i],ns[j]), dtype='d')
            else:
                block = np.ones((ns[i],ns[j]), dtype='d')*it*0.5
                it+=1

                Crosscovs[i][j] = C.Points(block, labels='Cross covariation %i %i'%(i, j))

            crosscovs[i][j] = block
            if i!=j:
                crosscovs[j][i] = block.T

    fulldata = np.concatenate(dataset)
    fullsyst = np.block(crosscovs)
    fullcovmat = np.diag(fulldata)+fullsyst

    Dataset = [C.Points(data, labels='Data %i'%i) for i, data in enumerate(dataset)]
    Statset = [C.Points(data, labels='Stat %i'%i) for i, data in enumerate(dataset)]

    Cp = C.CovariatedPrediction(labels=['Concatenated prediction', 'Base covariance matrix', 'Full cov mat Cholesky decomposition'])

    for Data, Stat in zip(Dataset, Statset):
        Cp.append(Data)
        Cp.covariate(Stat, Data, 1, Data, 1)
    for i in range(len(ns)):
        for j in range(i):
            Cp.covariate(Crosscovs[i][j], Dataset[i], 1, Dataset[j], 1)
    Cp.covbase.covbase >> Cp.cov.covbase
    Cp.finalize()

    Cp.printtransformations()

    suffix = 'covariated_prediction'

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='X', ylabel='Y', title='Covariance matrix base')
    ax.minorticks_on()
    ax.grid()
    Cp.covbase.covbase.plot_matshow(colorbar=True, mask=0.0)
    path = os.path.join(str(tmp_path), suffix+'_covbase.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    path = os.path.join(str(tmp_path), suffix+'_graph.png')
    savegraph([Cp.prediction,Cp.covbase.covbase], path, verbose=False)
    allure_attach_file(path)

    data_o    = Cp.prediction.prediction.data()
    covbase_o = Cp.covbase.covbase.data()
    L_o       = np.tril(Cp.cov.L.data())
    L_expect = np.linalg.cholesky(fullcovmat)

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='X', ylabel='Y', title='L: covariance matrix decomposition')
    ax.minorticks_on()
    ax.grid()
    c=plt.matshow(np.ma.array(L_o,mask=L_o==0.0), fignum=False)
    add_colorbar(c)
    path = os.path.join(str(tmp_path), suffix+'_L.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    plt.close()

    assert (fulldata==data_o).all()
    assert (fullcovmat==covbase_o).all()
    assert np.allclose(L_o, L_expect)
