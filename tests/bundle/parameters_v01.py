#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict
from gna.env import env

cfg = NestedDict(
    bundle = dict( name='parameters',
                   version='v01',
                   nidx=[
                       ('s', 'source',   ['SA', 'SB']),
                       'name',
                       ('d', 'detector', ['D1', 'D2', 'D3']),
                       # ('e', 'element', ['e0', 'e1'])
                       ],
                   # major=('s', 'd')
                   ),
    parameter = 'rate0',
    # label='Flux normalization {source}->{detector} ({element})',
    label='Flux normalization {source}->{detector}',
    # separate_uncertainty = 'norm0',
    pars = uncertaindict(
        [
            ( 'SA.D1', 1.0 ),
            ( 'SB.D1', 2.0 ),
            ( 'SA.D2', 3.0 ),
            ( 'SB.D2', 4.0 ),
            ( 'SA.D3', 5.0 ),
            ( 'SB.D3', 6.0 ),
            ( 'SA.D4', 7.0 ),
            ( 'SB.D4', 8.0 ),
            ],
        uncertainty = 1.0,
        mode = 'percent',
        ),
)

b, = execute_bundle(cfg=cfg)

env.globalns.printparameters(labels=True)
