#!/usr/bin/env python

from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundles
from gna.env import env
from gna.configurator import NestedDict, uncertain

cfg = NestedDict(
    bundle = 'dummy',
    name = 'dymmubundle',
    indices = [
        ('n', 'num',   ['1', '2', '3']),
        ('a', 'alph',  ['a', 'b', 'c']),
        ('z', 'zyx',   ['X', 'Y', 'Z'])
        ],
    format = 'var.{num}.{alph}.{zyx}',
    input = True,
    size = 10,
    debug = True
    )

ns = env.globalns
shared = NestedDict()
b, = execute_bundles( common_namespace=ns, cfg=cfg, shared=shared )

print( shared )
