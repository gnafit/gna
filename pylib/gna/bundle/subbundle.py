# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class subbundle(TransformationBundle):
    bundle = None
    def __init__(self, cfgkey, **kwargs):
        self.cfgkey=cfgkey
        kwargs['cfg'] = kwargs['cfg'][cfgkey]
        super(subbundle, self).__init__( **kwargs )

    def build(self):
        self.bundles = execute_bundle(cfg=self.cfg, shared=self.shared)

        if len(self.bundles)==1:
            self.bundle = bundle = self.bundles[0]
            self.objects             = bundle.objects
            self.transformations_in  = bundle.transformations_in
            self.transformations_out = bundle.transformations_out
            self.outputs             = bundle.outputs
            self.inputs              = bundle.inputs

    def __getattr__(self, attr):
        if self.bundle is None:
            raise Exception('Can not forward attributes to multiple bundles')

        return getattr(self.bundle, attr)
