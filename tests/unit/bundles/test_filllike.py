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

def test_filllike_v01(tmp_path):
    x1 = C.Points(np.linspace(1.0, 6.0, 6), labels='Input 1')

    expr = [
            'TwosLike| x1()',
            ]
    a =  Expression_v01(expr)
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    cfg = NestedDict(
            filllike = dict(
                    bundle = dict(name='filllike', version='v01', major=()),
                    instances = {
                        'TwosLike': None,
                        },
                    value = 2.0
                    ),
            x1 = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'x1',
                inputs = None,
                outputs = x1.single()
                ),
            )
    context = ExpressionContext_v01(cfg, ns=env.globalns('test_filllike_v01'))
    a.build(context)

    path = tmp_path/'filllike_graph.pdf'
    print(path)
    savegraph(context.outputs.TwosLike, str(path), verbose=False)
    allure_attach_file(path)
