# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.bundle import *

class concatenate(TransformationBundle):
    def __init__(self, **kwargs):
        super(concatenate, self).__init__( **kwargs )
        self.transformations_in = self.transformations_out

    def build(self):
        concat = R.Prediction(ns=self.common_namespace)
        for ns in self.namespaces:
            """Create and store the input"""
            self.inputs[ns.name] = concat.append(ns.name)

        self.objects['concat']                               = concat
        self.transformations_out[self.common_namespace.name] = concat.prediction
        self.outputs[self.common_namespace.name]             = concat.prediction.prediction

        """Define observables"""
        obsname = self.cfg.get( 'observable', None )
        if obsname:
            self.common_namespace.addobservable(obsname, concat.prediction.prediction, ignorecheck=True)


