# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict

from gna.bundle import *
from gna.bundle.connections import each_pair

class bundlechain_v01(TransformationBundle):
    name = 'detector'
    def __init__(self, edges, **kwargs):
        super(bundlechain_v01, self).__init__( **kwargs )

        self.edges=edges

        self.bundles = OrderedDict()

    def build(self):
        args = dict( namespaces=self.namespaces,
                     common_namespace=self.common_namespace,
                     storage=self.storage,
                     edges=self.edges )

        for bundlename in self.cfg.chain:
            self.bundles[bundlename] = execute_bundle( cfg=self.cfg[bundlename], **args )

        for b1, b2 in each_pair( self.bundles.values() ):
            # print( 'Connect {b1}.{output}->{b2}.{input} ({count})'.format( b1=b1.name, output=b1.outputs[0].name(),
                                                                           # b2=b2.name, input=b2.inputs[0].name(),
                                                                           # count=len(b1.inputs) ) )
            for output, input in zip( b1.outputs, b2.inputs ):
                input( output )

        self.output_transformations = self.bundles.values()[-1].output_transformations
        self.inputs, self.output = self.bundles.values()[0].inputs, self.bundles.values()[-1].outputs

