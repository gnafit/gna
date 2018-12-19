#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.bundle import execute_bundle
from gna.configurator import NestedDict

cfg = NestedDict(
    bundle = dict( name='parameters',
                   version='v01',
                   nidx=[ ('i', 'index', ['1', '2', '3']) ]
                   ),
    )

b, = execute_bundle(cfg=cfg)
