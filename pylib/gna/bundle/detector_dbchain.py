# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.env import env, namespace

from gna.bundle import *
from gna.bundle import declare_all

@declare_bundle('dbchain_v01')
class detector_dbchain(TransformationBundle):
    name = 'detector'
    def __init__(self, edges, **kwargs):
        super(detector_dbchain, self).__init__( **kwargs )

        self.edges=edges

        self.bundles = dict()
        self.connections = dict(
                iav = dict(
                    input=( 'smear', 'Ntrue' ),
                    output=('smear', 'Nvis')
                    ),
                nonlinearity = dict(
                    input=( 'smear', 'Ntrue' ),
                    output=('smear', 'Nvis')
                ),
                eres = dict(
                    input=( 'smear', 'Nvis' ),
                    output=('smear', 'Nrec')
                ),
                rebin = dict(
                    input=( 'rebin', 'histin' ),
                    output=('rebin', 'histout')
                )
            )

    def build(self):
        args = dict( namespaces=self.namespaces,
                     common_namespace=self.common_namespace,
                     storage=self.storage,
                     edges=self.edges )

        self.lists = ()
        connections = []
        bundlename_p = None
        for bundlename in self.cfg.chain:
            _, bundle = execute_bundle( cfg=self.cfg[bundlename], **args )
            self.bundles[bundlename] = bundle
            self.lists+=bundle.output,

            if bundlename_p:
                connections.append( (self.connections[bundlename_p]['output'], self.connections[bundlename]['input']) )
            bundlename_p = bundlename

        self.inputs, self.output = self.lists[0], self.lists[-1]

        transformations_map( self.lists, connections )

        return self.output


