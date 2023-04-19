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

def test_switch_v01(tmp_path):
    x1 = C.Points(np.linspace(1.0, 6.0, 6), labels='Input 1')
    x2 = C.Points(np.linspace(2.0, 7.0, 6), labels='Input 2')
    x3 = C.Points(np.linspace(3.0, 8.0, 6), labels='Input 3')

    expr = [
            'Switch| x1(), x2(), x3()',
            ]
    a =  Expression_v01(expr)
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    cfg = NestedDict(
            switch = dict(
                    bundle = dict(name='switch', version='v01', major=()),
                    instances = {
                        'Switch': 'Switch',
                        },
                    varlabel = 'switch',
                    ninputs = 3
                    ),
            x1 = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'x1',
                inputs = None,
                outputs = x1.single()
                ),
            x2 = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'x2',
                inputs = None,
                outputs = x2.single()
                ),
            x3 = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'x3',
                inputs = None,
                outputs = x3.single()
                ),
            )
    context = ExpressionContext_v01(cfg, ns=env.globalns('test_switch_v01'))
    a.build(context)

    path = tmp_path/'switch_graph.pdf'
    print(path)
    savegraph(context.outputs.Switch, str(path), verbose=False, namespace=context.namespace())
    allure_attach_file(path)
