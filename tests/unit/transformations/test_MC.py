#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig
import pytest
from gna.unittest import allure_attach_file, savegraph
from gna import constructors as C
from gan.bindings import common

@pytest.mark.parametrize('mc', ['Snapshot']) #, 'PoissonToyMC', 'NormalStatsToyMC', 'NormalToyMC', 'CovarianceToyMC'])
@pytest.mark.parametrize('data', [10.0, 100.0, 1000.0])
def test_mc(mc, data):
    print(mc, data)
    size = 10
    data1 = N.ones(size, dtype='d')*data
    data2 = N.arange(size, dtype='d')*data
    data3 = (size - N.arange(size, dtype='d'))*data

    data_v = (data1, data2, data3)
    err_v  = tuple(data**0.5 for data in data_v)
    hists  = tuple(C.Histogram(N.arange(data.size+1, dtype='d'), data) for data in data_v)
    errs   = tuple(C.Points(err) for err in err_v)

    if mc=='NormalToyMC':
        inputs = tuple(zip(hists, errs))
    elif mc=='CovarianceToyMC':
        inputs = tuple(zip(hists, errs))
    else:
        inputs = hists

    MCclass = getattr(C, mc)
    mcobject = MCclass()

    for inp in inputs:
        mcobject.add_input(inp.single())

    mcobject.printtransformations()

    for out in mcobject.trans
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( '' )
    ax.set_ylabel( '' )
    ax.set_title( '' )


