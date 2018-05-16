#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundle
from gna.env import env
from gna.configurator import NestedDict, uncertain
from collections import OrderedDict

cfg = NestedDict(
    bundle = 'dummy',
    indices = [
        ('n', 'num',   ['1', '2', '3']),
        ('a', 'alph',  ['a', 'b', 'c']),
        ('z', 'zyx',   ['X', 'Y', 'Z'])
        ],
    format = 'var.{num}.{alph}.{zyx}',
    input = True,
    size = 10
    )

ns = env.globalns
shared = NestedDict()
b, = execute_bundle( common_namespace=ns, cfg=cfg, shared=shared )

print( shared )
