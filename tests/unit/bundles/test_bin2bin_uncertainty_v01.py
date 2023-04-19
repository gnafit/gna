#!/usr/bin/env python3

import numpy as np
from load import ROOT as R
from gna.configurator import NestedDict
from gna.bundle import execute_bundles
from gna.env import env
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.expression.index import NIndex
import gna.constructors as C
from gna.unittest import allure_attach_file, savegraph
import os

def test_bin2bin_uncertainty_v01(tmp_path):
    expr = [
            'uncrt()',
            ]
    a =  Expression_v01(expr)
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    cfg = NestedDict(
            uncrt = dict(
                    bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                    name = 'uncrt',
                    mode = 'relative',
                    edges_target = np.arange(0.0, 10.0, 0.5),
                    uncertainty={
                        'uncertainty': 0.5,
                        'binwidth': 1.0
                        }
                    ),
            )
    context = ExpressionContext_v01(cfg, ns=env.globalns('test_bin2bin_uncertainty_v01'))
    a.build(context)

    bundle = context.required_bundles['uncrt'].bundle
    assert len(bundle.parameters)==cfg['uncrt']['edges_target'].size-1
    par0=bundle.parameters[0]
    checkval = 0.5 * (1/0.5)**0.5
    assert par0.sigma()==checkval

    path = tmp_path/'bin2bin_uncertainty_v01_graph.pdf'
    print(path)
    savegraph(context.outputs.uncrt, str(path), verbose=False, namespace=context.namespace())
    allure_attach_file(path)
