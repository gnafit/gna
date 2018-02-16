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
        self.concat = R.Prediction(ns=self.common_namespace)
        for ns in self.namespaces:
            self.concat.append()
            concatenate = C.Rebin( self.cfg.edges, self.cfg.rounding, ns=ns )

        """Save the transformations"""
        self.objects[('rebin', ns.name)]  = rebin
        self.transformations_out[ns.name] = rebin.rebin
        self.inputs[ns.name]              = rebin.rebin.histin
        self.outputs[ns.name]             = rebin.rebin.histout

        """Define observables"""
        ns.addobservable('rebin', rebin.rebin.histout, ignorecheck=True)


