# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class bundlesum_v01(TransformationBundle):
    def __init__(self, **kwargs):
        super(bundlesum_v01, self).__init__( **kwargs )

        self.bundles = NestedDict()

    def build(self):
        args = dict( namespaces=self.namespaces,
                     common_namespace=self.common_namespace,
                     edges=self.edges )

        for bundlename in self.cfg.chain:
            self.bundles[bundlename], = execute_bundle( cfg=self.cfg[bundlename], **args )


        import IPython
        IPython.embed()
        osum = R.Sum()
        self.objects['sum'] = osum
        for name, bundle in self.bundles.items() :
            # print( 'Connect {b1}.{output}->{b2}.{input} ({count})'.format( b1=type(b1).__name__, output=b1.outputs[0].name(),
                                                                           # b2=type(b2).__name__, input=b2.inputs[0].name(),
                                                                           # count=len(b1.inputs) ) )
            if len( bundle.outputs )!=1:
                raise Exception('bundlesum_v01 can sum only bundles with single output. Exception on '+name)
            osum.add(bundle.outputs[0])

        """Save transformations"""
        self.transformations_out['sum'] = osum
        self.outputs                   = self.bundles.values()[-1].outputs
