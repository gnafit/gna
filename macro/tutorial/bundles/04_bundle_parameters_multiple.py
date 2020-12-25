#!/usr/bin/env python

import load
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna.env import env

#
# List of parameters 1
#
cfg1 = NestedDict(
    bundle = dict(
        # Bundle name
        name='parameters',
        # Bundle version
        version='ex02',
        # Multi-index specification (may be set outside)
        nidx=[
            ('s', 'source',   ['SA', 'SB']),
            'name',
            ('d', 'detector', ['D1', 'D2', 'D3']),
            ],
        ),
    # Parameter name to be defined
    parameter = 'rate0',
    # Label format
    label='Flux normalization {source}->{detector}',
    # Dictionary with parameter values and uncertainties
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
b1 = execute_bundle(cfg1, namespace=env.globalns('bundle1'))
env.globalns('bundle1').printparameters(labels=True); print()

#
# List of parameters 2
#
cfg2 = NestedDict(
    bundle = dict( name='parameters',
                   version='ex02',
                   nidx=[
                       ('s', 'source',   ['SA', 'SB']),
                       ('d', 'detector', ['D1', 'D2', 'D3']),
                       ('e', 'element', ['e0', 'e1'])
                       ],
                   major=('s', 'd')
                   ),
    parameter = 'rate1',
    label='Flux normalization {source}->{detector} ({element})',
    pars = uncertaindict(
        [
            ( 'SA.D1', (1.0, 1.0) ),
            ( 'SB.D1', (2.0, 2.0) ),
            ( 'SA.D2', (3.0, 3.0) ),
            ( 'SB.D2', (4.0, 4.0) ),
            ( 'SA.D3', (5.0, 5.0) ),
            ( 'SB.D3', (6.0, 6.0) ),
            ( 'SA.D4', (7.0, 7.0) ),
            ( 'SB.D4', (8.0, 8.0) ),
            ],
        mode = 'percent',
        ),
)
b2 = execute_bundle(cfg2, namespace=env.globalns('bundle2'))
env.globalns('bundle2').printparameters(labels=True); print()

#
# List of parameters 3 (0d)
#
cfg3 = NestedDict(
    bundle = dict( name='parameters',
                   version='ex02',
                   ),
    parameter = 'constant',
    label='some constant',
    pars = uncertain( -1.0, uncertainty=4.0, mode='percent')
)
b3 = execute_bundle(cfg3, namespace=env.globalns('bundle3'))
env.globalns.printparameters(labels=True)
