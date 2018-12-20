# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class rebin(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        super(rebin, self).__init__( *args, **kwargs )
        self.transformations_in = self.transformations_out

    def build(self):
        for ns in self.namespaces:
            rebin = C.Rebin( self.cfg.edges, self.cfg.rounding, ns=ns )

            """Save the transformations"""
            self.objects[('rebin', ns.name)]  = rebin
            self.transformations_out[ns.name] = rebin.rebin
            self.inputs[ns.name]              = rebin.rebin.histin
            self.outputs[ns.name]             = rebin.rebin.histout

            """Define observables"""
            self.addcfgobservable(ns, rebin.rebin.histout, 'rebin', ignorecheck=True)


