# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class bundlelist_v01(TransformationBundle):
    def __init__(self, listkey='bundles_list', **kwargs):
        self.listkey=listkey
        super(bundlelist_v01, self).__init__( **kwargs )

        self.bundles = NestedDict()

    def build(self):

        bundlelist = self.cfg.get(self.listkey, None)
        if not bundlelist:
            raise Exception('Bundle list is not provided (key: {})'.format(self.listkey))
        for bundlename in bundlelist:
            self.bundles[bundlename], = execute_bundle( cfg=self.cfg[bundlename], shared=self.shared )

        if len(self.bundles)==1:
            bundle = self.bundles.values()[0]
            self.objects             = bundle.objects
            self.transformations_in  = bundle.transformations_in
            self.transformations_out = bundle.transformations_out
            self.outputs             = bundle.outputs
            self.inputs              = bundle.inputs
