# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class bundleproduct_v01(TransformationBundle):
    def __init__(self, listkey='bundleproduct_list', **kwargs):
        self.listkey=listkey
        super(bundleproduct_v01, self).__init__( **kwargs )

        self.bundles = NestedDict()

    def build(self):
        bundlelist = self.cfg.get(self.listkey, None)
        if not bundlelist:
            raise Exception('Bundle list is not provided (key: {})'.format(self.listkey))
        for bundlename in bundlelist:
            self.bundles[bundlename], = execute_bundle( cfg=self.cfg[bundlename], shared=self.shared )

        debug = self.cfg.get('debug', False)
        names=self.bundles.values()[0].outputs.keys()

        chaininput = self.cfg.get('chaininput', None)
        if chaininput and not isinstance(chaininput, str):
            raise Exception( 'chaininput should be a string' )
        for name in names:
            ns = self.common_namespace(name)
            prod = R.Product(ns=ns)

            if chaininput:
                inp = prod.multiply(chaininput)
                """Save unconnected input"""
                self.inputs[name]             = inp
                self.transformations_in[name] = prod

            for bundlename, bundle in self.bundles.items():
                if not name in bundle.outputs:
                    raise Exception( 'Failed to find output for {} in {} {}'.format( name, type(bundle).__name__, bundlename ) )

                if debug:
                    print( '    add {} ({}) {}'.format(bundlename, type(bundle).__name__, bundle.outputs[name].name()) )
                prod.add(bundle.outputs[name])

            """Save transformations"""
            self.objects[('product', name)]= prod
            self.transformations_out[name] = prod.product
            self.outputs[name]             = prod.product.product

            """Define observable"""
            self.addcfgobservable(ns, prod.product.product, ignorecheck=bool(chaininput))

