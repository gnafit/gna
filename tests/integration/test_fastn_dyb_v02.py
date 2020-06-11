#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
import pytest
import allure
from collections import OrderedDict

from load import ROOT as R

from mpl_tools.helpers import savefig
from gna.configurator import NestedDict, uncertaindict
from gna.bundle import execute_bundles
from gna.unittest import allure_attach_file, savegraph
from gna.env import env
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.expression.index import NIndex
import gna.constructors as C


def test_snf_spectrum_expression(tmp_path):
    """Test for new version of Daya Bay fast neutron background bundle,
    updated for new integrators"""

    indices = [
            ('s', 'site',        ['EH1', 'EH2', 'EH3']),
            ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34'],
                                 dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')), ('EH3', ('AD31', 'AD32', 'AD33', 'AD34'))]))),
        ]

    expr = ['evis_edges()',
            'fastn_shape[s]',
            'bkg_spectrum_fastn[s]()'
            ]
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    lib = dict()
    a.guessname(lib, save=True)

    ns = env.globalns('fastn')
    cfg = NestedDict(
            integral = NestedDict(
                bundle   = dict(name='integral_2d1d', version='v03'),
                variables = ('evis', 'ctheta'),
                edges    = np.linspace(0.0, 12.0, 241, dtype='d'),
                xorders   = 4,
                yorder   = 2,
                ),
            bkg_spectrum_fastn=NestedDict(
                    bundle=dict(name='dayabay_fastn_power', version='v02', major='s'),
                    parameter='fastn_shape',
                    name='bkg_spectrum_fastn',
                    normalize=(0.7, 12.0),
                    bins='evis_edges',
                    order=2,
                    ),
            fastn_shape=NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter='fastn_shape',
                    label='Fast neutron shape parameter for {site}',
                    pars=uncertaindict(
                        [ ('EH1', (67.79, 0.1132)),
                          ('EH2', (58.30, 0.0817)),
                          ('EH3', (68.02, 0.0997)) ],
                        mode='relative',
                        ),
                    ),
            )
    context = ExpressionContext_v01(cfg, ns=ns)
    a.build(context)
