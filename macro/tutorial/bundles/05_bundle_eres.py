#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna.env import env

cfg = NestedDict(
    bundle = dict(
        name='detector_eres',
        version='ex01',
        nidx=[ ('d', 'detector', ['D1', 'D2', 'D3']) ],
        major=[],
        ),
    # label='Flux normalization {source}->{detector}',
    parameter = 'eres',
    pars = uncertaindict(
        [('a', 0.014764) ,
         ('b', 0.0869) ,
         ('c', 0.0271)],
        mode='percent',
        uncertainty=30
        ),
)
b = execute_bundle(cfg)
env.globalns.printparameters(labels=True); print()

