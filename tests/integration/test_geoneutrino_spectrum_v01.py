#!/usr/bin/env python

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
                data   = 'data/data-common/geo-neutrino/2006-sanshiro/geoneutrino-luminosity_{isotope}_truncated.knt'
            )
    ns = env.globalns('geonu')

    geonu, = execute_bundles(cfg, namespace=ns)
    ns.printparameters(labels=True)

    Enu >> geonu.context.inputs.values(nested=True)

    # Dump some info
    print(geonu.context.inputs)
    print(geonu.context.outputs)
    geonu.interp.values()[0].printtransformations()
    geonu.interp.values()[1].printtransformations()

    # Plot figures and graphs
    fig = plt.figure()
    ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel='N/MeV/s', title='Geo-neutrino luminosity (truncated at 1.7 MeV)')
    ax.minorticks_on()
    ax.grid()

    for k, v in geonu.context.outputs.items():
        ax.plot(_enu, v.data(), label=k)

    ax.legend()
    plt.show()

    savefig(os.path.join(str(tmp_path), '_spectra.png'))
    savegraph(Enu, os.path.join(str(tmp_path), '_graph.png'))

    ns.printparameters(labels=True)

