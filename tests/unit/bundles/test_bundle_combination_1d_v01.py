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
import pytest

def test_bundle_combination_1d_v01(tmp_path):
    indices = [
         ('a', 'major', ['a', 'b', 'c', 'd']),
         ('z', 'minor', ['x', 'y', 'z']),
        ]

    expr = [ 'testpar[a,z]' ]
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    print(a.expressions)
    a.guessname({}, save=True)

    cfg = NestedDict(
            parameters = dict(
                bundle = dict(name="bundles_combination_1d", version = "v01", major='a'),
                slices = {
                    'left': ('a', 'b'),
                    'right': 'rest'
                    },
                bundles = dict(
                    left = dict(
                        pars = dict(
                            bundle = dict(name="parameters", version = "v06", major='a'),
                            pars = {
                                'testpar': {
                                    'a':  1,
                                    'b':  2,
                                    },
                                'uncertainty': 1,
                                'uncertainty_mode': 'percent',
                                'meta': { 'labels': {'testpar': 'Left parameter major={major}, minor={minor}'} }
                                },
                            separate_uncertainty = '{}_scale',
                            objectize = True
                            )
                        ),
                    right = dict(
                        pars = dict(
                            bundle = dict(name="parameters", version = "v06", major='a'),
                            pars = {
                                'testpar': {
                                    'c':  3,
                                    'd':  4,
                                    },
                                'meta': { 'labels': {'testpar': 'Right parameter major={major}, minor={minor}'} }
                                },
                            state = 'fixed',
                            objectize = True
                            )
                        )
                    ),
                ),
            )
    ns=env.globalns('test_bundle_combination_1d_v01')
    context = ExpressionContext_v01(cfg, ns=ns)
    a.build(context)

    ns.printparameters(labels=True)

    print('Outputs:')
    print(context.outputs.__str__(nested=True))
    print()

