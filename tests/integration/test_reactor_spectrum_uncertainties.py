#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import os
import pytest
import allure

from load import ROOT as R

from mpl_tools.helpers import savefig
from gna.configurator import NestedDict
from gna.bundle import execute_bundles
from gna.unittest import allure_attach_file, savegraph
from gna.env import env
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.expression.index import NIndex
import gna.constructors as C


def test_reactor_spectrum_unc_v01(tmp_path):

    _enu = np.linspace(1.8, 10.0, 500, dtype='d')
    Enu = C.Points(_enu, labels='anue energy')

    indices = [
         ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241']),
         ('r', 'reactor', ['DB1', 'LA1'])
        ]


    expr = ['anuspec[i](enu())', 'offeq_correction[i,r]| enu(), anuspec[i]()']
    expr.append("corrected_spectrum[i,r]|enu(), anuspec[i]()")
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    ns_anuexpr = env.globalns('anue_expr')
    cfg = NestedDict(
            anuspec = NestedDict(
                bundle = dict(name='reactor_anu_spectra', version='v04'),
                name = 'anuspec',
                filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                            'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                varmode='log',
                varname='anu_weight_{index}',
                free_params=True,
                ns_name='spectral_weights',
                edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
                ),
            enu = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'enu',
                inputs = None,
                outputs = Enu.single(),
                ),
            corrected_spectrum = NestedDict(
                bundle = dict(name='huber_mueller_spectra_uncertainty', version='v01', major='ir'),
                reac_idx='r',
                iso_idx='i',
                ns_name='hm_uncertainties',
                files_corr=['output/anue_spectra/Huber_corrrelated_unc_extrap_{isotope}_13.0_0.05_MeV.txt',
                            'output/anue_spectra/Mueller_corrrelated_unc_extrap_{isotope}_13.0_0.05_MeV.txt'],
                files_uncorr=['output/anue_spectra/Huber_uncorrrelated_unc_extrap_{isotope}_13.0_0.05_MeV.txt',
                              'output/anue_spectra/Mueller_uncorrrelated_unc_extrap_{isotope}_13.0_0.05_MeV.txt'],
                ),
            offeq_correction = NestedDict(
                bundle = dict(name='reactor_offeq_spectra',
                              version='v04', major='ir'),
                offeq_data = './data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
                ),
            )
    context = ExpressionContext_v01(cfg, ns=ns_anuexpr)
    a.build(context)
    ns_anuexpr.printparameters(labels=True)
    u235_spec = context.outputs.corrected_spectrum.U235.DB1
    suffix = 'hm_uncertainty_corrected_spec'
    #  savefig(path.replace('.png','.pdf'), dpi=300)

    path = os.path.join(str(tmp_path), suffix + '_graph.png')
    savegraph(u235_spec, path)
    allure_attach_file(path)
    path = os.path.join(str(tmp_path), suffix + '_graph.pdf')
    savegraph(u235_spec, path)
    allure_attach_file(path)
