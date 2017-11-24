# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from gna.bundle import *

@declare_bundle('rebin')
class bundle_rebin(TransformationBundle):
    name = 'rebin'
    def __init__(self, **kwargs):
        super(bundle_rebin, self).__init__( **kwargs )

    def build(self):
        self.output=()
        edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        for ns in self.namespaces:
            with ns:
                eres = R.Rebin( edges.size, edges, int( self.cfg.rounding ) )
                self.output+=eres,

        return self.output

