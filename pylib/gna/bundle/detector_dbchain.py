# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.env import env, namespace

from gna.bundle import *
from gna.bundle import declare_all
from gna.bundle.detector_iav import detector_iav_from_file
from gna.bundle.detector_nl import detector_nl_from_file

@declare_bundle('dbchain_v01')
class detector_dbchain(TransformationBundle):
    def __init__(self, edges, **kwargs):
        kwargs.setdefault( 'storage_name', 'detector')
        super(detector_dbchain, self).__init__( **kwargs )

        iavlist, _ = detector_iav_from_file( namespaces=self.namespaces, storage=self.storage, **self.cfg.detector.iav )
        nllist, _  = detector_nl_from_file( edges=edges, namespaces=self.namespaces, storage=self.storage, **self.cfg.detector.nonlinearity )

        reslist = transformations_map( iavlist, 'aaa', nllist, 'aaa' )

        # return reslist, storage


