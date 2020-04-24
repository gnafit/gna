#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
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


def test_geoneutrino_spectrum_v01(tmp_path):
    _enu = np.arange(1., 8.0+1.e-6, 0.01, dtype='d')
    Enu = C.Points(_enu, labels='anue energy')

    cfg = NestedDict(
                bundle = dict(name='geoneutrino_spectrum', version='v01'),
                data   = 'data/data-common/geo-neutrino/2006-sanshiro/AntineutrinoSpectrum_{isotope}.knt'
            )
    ns = env.globalns('geonu')

    geonu, = execute_bundles(cfg, namespace=ns)
    ns.printparameters(labels=True)

    # for iso in offeq.context.inputs.offeq_correction.values():
        # try:
            # for _input in iso.values():
                # Enu >> _input.values()
        # except AttributeError:
                # Enu >> iso

    # fig, ax = plt.subplots()
    # for iso in indices[0][2]:
        # corrected_spectra = offeq.context.outputs.offeq_correction[iso]['DB1'].data()
        # ax.plot(Enu.single().data(), corrected_spectra -1., label=iso)
    # ax.set_title("Offequilibrium correction")
    # ax.grid()
    # ax.legend(loc='best')
    # ax.set_xlabel("Antineutrino energy, MeV")
    # ax.set_ylabel("(Corrected - nominal) / nominal")

    # suffix = 'correction'
    # path = os.path.join(str(tmp_path), suffix + '.png')
    # savefig(path, dpi=300)
    # savefig(path.replace('.png','.pdf'), dpi=300)
    # allure_attach_file(path)

    # path = os.path.join(str(tmp_path), suffix + '_graph.png')
    # savegraph(offeq.context.outputs.offeq_correction['U235']['DB1'], path)
    # allure_attach_file(path)

# def test_offeq_correction_expression(tmp_path):
    # """Same test but build bundle with expressions"""

    # _enu = np.linspace(1., 8.0, 500, dtype='d')
    # Enu = C.Points(_enu, labels='anue energy')

    # indices = [
         # ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241']),
         # ('r', 'reactor', ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4'])
        # ]


    # expr = ['offeq_correction[i,r](enu())']
    # a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    # a.parse()
    # lib = dict()
    # a.guessname(lib, save=True)

    # ns_offeq = env.globalns('offeq_expr')
    # cfg = NestedDict(
            # offeq_correction = NestedDict(
                # bundle = dict(name='reactor_offeq_spectra',
                              # version='v03', major='ir'),
                # offeq_data = './data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
                # ),
            # enu = NestedDict(
                # bundle = NestedDict(name='predefined', version='v01', major=''),
                # name = 'enu',
                # inputs = None,
                # outputs = Enu.single(),
                # ),
            # )
    # context = ExpressionContext_v01(cfg, ns=ns_offeq)
    # a.build(context)
    # ns_offeq.printparameters(labels=True)


    # fig, ax = plt.subplots()
    # for iso in indices[0][2]:
        # corrected_spectra = context.outputs.offeq_correction[iso]['DB1'].data()
        # ax.plot(Enu.single().data(), corrected_spectra -1., label=iso)
    # ax.set_title("Offequilibrium correction")
    # ax.legend(loc='best')
    # ax.grid()
    # ax.set_xlabel("Antineutrino energy, MeV")
    # ax.set_ylabel("(Corrected - nominal) / nominal")

    # suffix = 'correction'
    # path = os.path.join(str(tmp_path), suffix + '.png')
    # savefig(path, dpi=300)
    # savefig(path.replace('.png','.pdf'), dpi=300)
    # allure_attach_file(path)

    # path = os.path.join(str(tmp_path), suffix + '_graph.png')
    # savegraph(context.outputs.offeq_correction['U235']['DB1'], path)

    # allure_attach_file(path)
