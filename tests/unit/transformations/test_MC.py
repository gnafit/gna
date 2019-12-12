#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig
import pytest
from gna.unittest import allure_attach_file, savegraph
from gna import constructors as C
from gna.bindings import common
import os

@pytest.mark.parametrize('scale', [0.1, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize('mc', ['Snapshot', 'PoissonToyMC', 'NormalStatsToyMC', 'NormalToyMC', 'CovarianceToyMC'])
def test_mc(mc, scale, tmp_path):
    size = 20
    data1 = np.ones(size, dtype='d')*scale
    data2 = (1.0+np.arange(size, dtype='d'))*scale
    data3 = (size - np.arange(size, dtype='d'))*scale

    data_v = (data1, data2, data3)
    err_v  = tuple(data**0.5 for data in data_v)
    covmat_stat_l_v = tuple(np.diag(err) for err in err_v)
    hists  = tuple(C.Histogram(np.arange(data.size+1, dtype='d'), data) for data in data_v)
    errs   = tuple(C.Points(err) for err in err_v)
    covmats_l = tuple(C.Points(l) for l in covmat_stat_l_v)

    if mc=='NormalToyMC':
        inputs = tuple(zip(hists, errs))
    elif mc=='CovarianceToyMC':
        inputs = tuple(zip(hists, covmats_l))
    else:
        inputs = tuple((hist,) for hist in hists)

    MCclass = getattr(C, mc)
    mcobject = MCclass()

    for inp in inputs:
        mcobject.add_inputs(*inp)

    mcobject.printtransformations()

    for i, (hist, out) in enumerate(zip(hists, mcobject.transformations[0].outputs.values())):
        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( '' )
        ax.set_ylabel( '' )
        ax.set_title('Check {}, input {}, scale {}'.format(mc, i, scale))

        hist.plot_hist(color='black', label='input')
        out.plot_errorbar(yerr='stat', linestyle='--', label='output')

        ax.legend()

        suffix = '_'.join((mc, str(int(scale)), str(i)))
        path = os.path.join(str(tmp_path), suffix+'.png')
        savefig(path, dpi=300)
        allure_attach_file(path)

        plt.close()

        data_mean = hist.single().data()
        data_mc = out()
        data_diff =  data_mc - data_mean
        if mc=='Snapshot':
            assert (data_diff==0.0).all()
        else:
            if mc!='PoissonToyMC':
                assert (data_diff!=0.0).all()

            fdata_diff = np.fabs(data_diff)
            err = err_v[i]

            nsigma = 4.0
            assert (fdata_diff<nsigma*err).all()

            sum  = data_diff.sum()
            fsum = np.fabs(sum)
            assert fsum < nsigma * data_mean.sum()**0.5

            reldiff = data_diff/err

            chi2 = (reldiff**2).sum()
            chi2_diff = chi2 - size
            assert chi2_diff < nsigma*(2.0*size)**0.5
