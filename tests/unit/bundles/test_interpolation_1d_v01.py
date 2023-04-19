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

def test_interpolation_1d_v01(tmp_path):
    Coarse = C.Points(np.linspace(1.0, 6.0, 6), labels='x (coarse)')
    Fine   = C.Points(np.linspace(1.0, 6.0, 11), labels='new x (fine)')

    indices = [
         ('a', 'major', ['A', 'B']),
         ('z', 'minor', ['1', '2', '3']),
        ]

    InA, InZ = {}, {}
    for i, k in enumerate(indices[0][2]):
        InA[k] = C.Points(Coarse.single().data()*(i+1), labels=f'Inp {k}')
    for i, k in enumerate(indices[1][2]):
        InZ[k] = C.Points(-Coarse.single().data()*(i+3), labels=f'Inp {k}')

    expr = [
            'interp_base| x(), newx()',
            'interp| arga[a]()*argz[z]()',
            ]
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    cfg = NestedDict(
            interp=dict(
                    bundle = dict(name='interpolation_1d', version='v01', major=''),
                    name='interp',
                    kind='linear',
                    strategy=('constant', 'extrapolate'),
                    label='Example',
                    labelfmt='Example {autoindex}',
                ),
            x = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'x',
                inputs = None,
                outputs = Coarse.single()
                ),
            newx = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'newx',
                inputs = None,
                outputs = Fine.single()
                ),
            inpa = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major='a'),
                name = 'arga',
                inputs = None,
                outputs = {k: v.single() for k, v in InA.items()}
                ),
            inpz = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major='z'),
                name = 'argz',
                inputs = None,
                outputs = {k: v.single() for k, v in InZ.items()}
                ),
            )
    context = ExpressionContext_v01(cfg, ns=env.globalns('test_interpolation_1d_v01'))
    a.build(context)

    path = tmp_path/'interp_graph.pdf'
    print(path)
    savegraph(context.outputs.interp.A['1'], str(path), verbose=False)
    allure_attach_file(path)
