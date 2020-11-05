#!/usr/bin/env python2

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


def test_anue_free_spectra(tmp_path):
    """ Test implementation of a model of antineutrino spectra with free
    parameters in exponential parametrization.
    """

    _enu = np.linspace(1.8, 8.0, 500, dtype='d')
    Enu = C.Points(_enu, labels='anue energy')

    indices = [
         ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241'])
        ]


    expr = ['anuspec[i,r](enu())']
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    lib = dict()
    a.guessname(lib, save=True)

    ns_anuexpr = env.globalns('anue_expr')
    cfg = NestedDict(
            anuspec = NestedDict(
                bundle = dict(name='reactor_anu_spectra', version='v04'),
                name = 'anuspec',
                filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                            'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                # strategy = dict( underflow='constant', overflow='extrapolate' ),
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
            )
    context = ExpressionContext_v01(cfg, ns=ns_anuexpr)
    a.build(context)
    ns_anuexpr.printparameters(labels=True)
    u235_spec = context.outputs.anuspec.U235
    u235_spec.plot_vs(Enu.single(), label='default pars')

    ns_anuexpr['spectral_weights.anu_weight_5'].set(0.3)
    ns_anuexpr['spectral_weights.anu_weight_7'].set(-0.3)

    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'text.usetex': True})
    u235_spec.plot_vs(Enu.single(), label='update pars')
    plt.yscale('log')
    plt.xlabel(r'$E_{\nu}$, MeV')
    plt.ylabel('Anue per MeV')
    plt.legend()
    plt.title('Antineutrino spectrum')
    path = os.path.join(str(tmp_path), 'anuspec.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
