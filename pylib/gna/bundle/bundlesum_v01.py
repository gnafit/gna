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
        args = dict(namespaces=self.namespaces, common_namespace=self.common_namespace)

        for bundlename in self.cfg.list:
            self.bundles[bundlename], = execute_bundle( cfg=self.cfg[bundlename], **args )

        names=self.bundles.values()[0].outputs.keys()
        for name in names:
            ns = self.common_namespace(name)
            osum = R.Sum(ns=ns)

            print('Sum bundles for', name)
            for bundlename, bundle in self.bundles.items():
                if not name in bundle.outputs:
                    raise Exception( 'Failed to find output for {} in {} {}'.format( name, type(bundle).__name__, bundlename ) )

                print( '    add {} ({}) {}'.format(bundlename, type(bundle).__name__, bundle.outputs[name].name()) )
                osum.add(bundle.outputs[name])

            """Save transformations"""
            self.objects[('sum', name)]    = osum
            self.transformations_out[name] = osum.sum
            self.outputs[name]             = osum.sum.outputs['sum']

            """Define observable"""
            obsname = self.cfg.get('observable', '')
            if obsname:
                ns.addobservable( obsname, osum.sum.outputs['sum'] )

