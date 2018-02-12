# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class bundlechain_v01(TransformationBundle):
    name = 'detector'
    def __init__(self, edges, **kwargs):
        super(bundlechain_v01, self).__init__( **kwargs )

        self.edges=edges
        self.bundles = NestedDict()

    def build(self):
        args = dict( namespaces=self.namespaces,
                     common_namespace=self.common_namespace,
                     edges=self.edges )

        for bundlename in self.cfg.chain:
            self.bundles[bundlename], = execute_bundle( cfg=self.cfg[bundlename], **args )

        for b1, b2 in pairwise( self.bundles.values() ):
            # print( 'Connect {b1}.{output}->{b2}.{input} ({count})'.format( b1=b1.name, output=b1.outputs[0].name(),
                                                                           # b2=b2.name, input=b2.inputs[0].name(),
                                                                           # count=len(b1.inputs) ) )
            for (oname, output), (iname, input) in zip( b1.outputs.items(), b2.inputs.items() ):
                if oname!=iname:
                    raise Exception('Trying to connect inconsistent ouput-input pair')
                input( output )

        """Save transformations"""
        self.transformations_in  = self.bundles.values()[ 0].transformations_in
        self.transformations_out = self.bundles.values()[-1].transformations_out
        self.inputs              = self.bundles.values()[ 0].inputs
        self.outputs             = self.bundles.values()[-1].outputs
