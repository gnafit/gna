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

def test_arrange_v01(tmp_path):
    x1 = C.Points(np.linspace(1.0, 6.0, 6), labels='Input 1')
    x2 = C.Points(np.linspace(2.0, 7.0, 6), labels='Input 2')
    x3 = C.Points(np.linspace(3.0, 8.0, 6), labels='Input 3')

    indices = [
         ('z', 'minor', ['a', 'b', 'c']),
        ]
    expr = [
            'arrangement[z]| x1(), x2(), x3()',
            ]
    a =  Expression_v01(expr, indices=NIndex.fromlist(indices))
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    cfg = NestedDict(
            arrangement = dict(
                    bundle = dict(name='arrange', version='v01'),
                    names = 'arrangement'
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
    context = ExpressionContext_v01(cfg)
    a.build(context)

    path = tmp_path/'arrange_graph.pdf'
    print(path)
    savegraph((context.outputs.x1, context.outputs.x2, context.outputs.x3), str(path), verbose=False)
    allure_attach_file(path)
