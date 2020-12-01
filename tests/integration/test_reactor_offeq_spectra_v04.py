#!/usr/bin/env python

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


def test_offeq_correction_expression(tmp_path):
    """Tests for new version of offequilibrium correction with snapshoting
    initial spectrum"""

    _enu = np.linspace(1., 8.0, 500, dtype='d')
    Enu = C.Points(_enu, labels='anue energy')

    indices = [
         ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241']),
         ('r', 'reactor', ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4'])
        ]

    expr = ['enu()','anuspec[i](enu())', 'offeq_correction[i,r]| enu(), anuspec[i]()']
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    lib = dict()
    a.guessname(lib, save=True)

    ns_offeq = env.globalns('offeq_expr_v04')
    cfg = NestedDict(
            offeq_correction = NestedDict(
                bundle = dict(name='reactor_offeq_spectra',
                              version='v04', major='ir'),
                offeq_data = './data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
                ),
            enu = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'enu',
                inputs = None,
                outputs = Enu.single(),
                ),
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
            )
    context = ExpressionContext_v01(cfg, ns=ns_offeq)
    a.build(context)
    ns_offeq.printparameters(labels=True)


    fig, ax = plt.subplots()
    for iso in indices[0][2]:
        corrected_spectra = context.outputs.offeq_correction[iso]['DB1'].data()
        initial_spectra = context.outputs.anuspec[iso].data()
        ax.plot(Enu.single().data(), (corrected_spectra - initial_spectra)/initial_spectra, label=iso)
    ax.set_title("Offequilibrium corrected spectra")
    ax.legend(loc='best')
    ax.grid()
    ax.set_xlabel("Antineutrino energy, MeV")
    ax.set_ylabel("(Corrected - nominal) / nominal")
    suffix = 'offeq'
    path = os.path.join(str(tmp_path), suffix+'.png')
    savefig(path.replace('.png','.pdf'), dpi=300)
    allure_attach_file(path)

    path = os.path.join(str(tmp_path), suffix + '_graph.png')
    savegraph(context.outputs.offeq_correction['U235']['DB1'], path)
    allure_attach_file(path)
