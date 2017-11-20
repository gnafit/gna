# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.env import env

from gna.bundle.detector_iav import detector_iav_from_file
from gna.bundle.detector_nl import detector_nl_from_file

def transformations_map( sources, targets ):
    pass

def detector_dbchain( *args, **kwargs ):
    gstorage = kwargs.pop( 'storage', None )
    if gstorage:
        storage = gstorage( 'detector_effects' )
    else:
        storage = namespace( None, 'detector_effects' )

    namespaces = kwargs.pop( 'namespaces', [env.globalns] )

    iavlist, _ = detector_iav_from_file( **kwargs.pop( 'iavopts' ) )
    nllist, _  = detector_nl_from_file( **kwargs.pop( 'nlopts' ) )

    reslist = transformations_map( iavlist, nllist )

    return reslist, storage


