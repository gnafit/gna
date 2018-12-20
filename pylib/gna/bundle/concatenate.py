# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.bundle import *

class concatenate(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        super(concatenate, self).__init__( *args, **kwargs )
        self.transformations_in = self.transformations_out

    def build(self):
        concat = R.Concat(ns=self.common_namespace)
        for ns in self.namespaces:
            """Create and store the input"""
            self.inputs[ns.name] = concat.append(ns.name)

        self.objects['concat']                               = concat
        self.transformations_out[self.common_namespace.name] = concat.prediction
        self.outputs[self.common_namespace.name]             = concat.prediction.prediction

        """Define observables"""
        self.addcfgobservable(self.common_namespace, concat.prediction.prediction, ignorecheck=True)


