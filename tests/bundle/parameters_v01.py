#!/usr/bin/env python

from load import ROOT as R
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna.env import env

cfg1 = NestedDict(
    bundle = dict( name='parameters',
                   version='v01',
                   nidx=[
                       ('s', 'source',   ['SA', 'SB']),
                       'name',
                       ('d', 'detector', ['D1', 'D2', 'D3']),
                       ],
                   ),
    parameter = 'rate0',
    label='Flux normalization {source}->{detector}',
    separate_uncertainty = 'norm0',
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

cfg2 = NestedDict(
    bundle = dict( name='parameters',
                   version='v01',
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

cfg3 = NestedDict(
    bundle = dict( name='parameters',
                   version='v01',
                   ),
    parameter = 'constant',
    label='some constant',
    pars = uncertain( -1.0, uncertainty=4.0, mode='percent')
)

b1 = execute_bundle(cfg1)
b2 = execute_bundle(cfg2)
b3 = execute_bundle(cfg3)

env.globalns.printparameters(labels=True)

print('Provides:')
print(b1.provides(cfg1))
print(b1.provides(cfg2))
print(b1.provides(cfg3))

