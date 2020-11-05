#!/usr/bin/env python

import load
from gna.bundle import execute_bundle
from gna.configurator import NestedDict, uncertaindict, uncertain
from gna.env import env

#
# Bundle configuration
#
cfg = NestedDict(
    bundle = dict(
        name='parameters',
        version='ex01',
        ),
    pars = uncertaindict(
        [
            ( 'par_a',   (1.0, 1.0,  'percent') ),
            ( 'par_b',   (2.0, 0.01, 'relative') ),
            ( 'par_c',   (3.0, 0.5,  'absolute') ),
            ( 'group.a', (1.0, 'free' ) ),
            ( 'group.b', (1.0, 'fixed', 'Labeled fixed parameter' ) )
            ],
        ),
)

#
# Execute bundle configuration
#
b1 = execute_bundle(cfg)

#
# Print the parameters
#
env.globalns.printparameters(labels=True)
