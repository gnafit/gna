# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class rebin(TransformationBundle):
    def __init__(self, **kwargs):
        super(rebin, self).__init__( **kwargs )

    def build(self):
        for ns in self.namespaces:
            rebin = C.Rebin( self.cfg.edges, self.cfg.rounding, ns=ns )

            self.transformations_in[ns.name]  = rebin
            self.transformations_out[ns.name] = rebin

            self.inputs[ns.name]  = rebin.rebin.histin
            self.outputs[ns.name] = rebin.rebin.histout


