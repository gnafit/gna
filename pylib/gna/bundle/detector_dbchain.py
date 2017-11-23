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

    def build(self):
        args = dict( namespaces=self.namespaces, storage=self.storage )

        iavlist, self.iav = execute_bundle( cfg=self.cfg.iav, **args )
        nllist, self.nl   = execute_bundle( edges=self.edges, cfg=self.cfg.nonlinearity, **args )

        self.inputs, self.outputs = iavlist, nllist

        connections = [ (( 'smear', 'Nvis' ), ( 'smear', 'Ntrue' )) ]
        transformations_map( (iavlist, nllist), connections )

        return nllist

