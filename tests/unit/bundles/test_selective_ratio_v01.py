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

def test_selective_ratio_1d_v01(tmp_path):
    a = np.linspace(1.0, 6.0, 6)
    b = np.full(6, 2.0)
    A = C.Points(a, labels='Inp A')

    indices = [
         ('y', 'major', ['A', 'B', 'C']),
         ('z', 'minor', ['1', '2', '3']),
        ]

    Bs = {}
    Boutputs = {}
    for i1, k1 in enumerate(indices[0][2]):
        for i2, k2 in enumerate(indices[1][2]):
            Bs.setdefault(k1, {})
            Boutputs.setdefault(k1, {})
            p = Bs[k1][k2] = C.Points(b*i2, labels=f'Inp B: {k1}.{k2}')
            Boutputs[k1][k2] = p.single()

    expr = [
            'selective_ratio| A(), B[y,z]()'
            ]
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    cfg = NestedDict(
            selective_ratio=dict(
                    bundle = dict(name='selective_ratio', version='v01', major='y'),
                    name='selective_ratio',
                    substring_skip='B',
                    labelfmt_ratio='RATIO {autoindex}',
                    labelfmt_view='VIEW {autoindex}',
                    broadcast = True
                ),
            inpa = dict(
                bundle = dict(name='predefined', version='v01'),
                name = 'A',
                inputs = None,
                outputs = A.single()
                ),
            inpb = dict(
                bundle = dict(name='predefined', version='v01', major='yz'),
                name = 'B',
                inputs = None,
                outputs = Boutputs
                ),
            )
    context = ExpressionContext_v01(cfg, ns=env.globalns('test_selective_ratio_1d_v01'))
    a.build(context)

    path = tmp_path/'selective_ratio_graph.pdf'
    print(path)
    savegraph(context.outputs.A, str(path), verbose=False)
    allure_attach_file(path)
