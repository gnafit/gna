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
    def __init__(self, listkey='bundlesum_list', **kwargs):
        self.listkey=listkey
        super(bundlesum_v01, self).__init__( **kwargs )

        self.bundles = NestedDict()

    def build(self):
        args = dict(namespaces=self.namespaces, common_namespace=self.common_namespace)

        bundlelist = self.cfg.get(self.listkey, None)
        if not bundlelist:
            raise Exception('Bundle list is not provided (key: {})'.format(self.listkey))
        for bundlename in bundlelist:
            self.bundles[bundlename], = execute_bundle( cfg=self.cfg[bundlename], **args )

        debug = self.cfg.get('debug', False)
        names=self.bundles.values()[0].outputs.keys()

        chaininput = self.cfg.get('chaininput', None)
        if chaininput and not isinstance(chaininput, str):
            raise Exception( 'chaininput should be a string' )
        for name in names:
            ns = self.common_namespace(name)
            osum = R.Sum(ns=ns)

            if chaininput:
                inp = osum.add(chaininput)
                """Save unconnected input"""
                self.inputs[name]             = inp
                self.transformations_in[name] = osum

            for bundlename, bundle in self.bundles.items():
                if not name in bundle.outputs:
                    raise Exception( 'Failed to find output for {} in {} {}'.format( name, type(bundle).__name__, bundlename ) )

                if debug:
                    print( '    add {} ({}) {}'.format(bundlename, type(bundle).__name__, bundle.outputs[name].name()) )
                data = bundle.outputs[name].data().sum()
                osum.add(bundle.outputs[name])

            """Save transformations"""
            self.objects[('sum', name)]    = osum
            self.transformations_out[name] = osum.sum
            self.outputs[name]             = osum.sum.outputs['sum']

            """Define observable"""
            self.addcfgobservable(ns, osum.sum.outputs['sum'], ignorecheck=bool(chaininput))

