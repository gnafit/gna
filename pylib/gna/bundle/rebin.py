# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class rebin(TransformationBundle):
    name = 'rebin'
    def __init__(self, **kwargs):
        super(rebin, self).__init__( **kwargs )

    def build(self):
        edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        for ns in self.namespaces:
            with ns:
                rebin = R.Rebin( edges.size, edges, int( self.cfg.rounding ) )
                self.output_transformations+=rebin,

                self.inputs  += rebin.rebin.histin,
                self.outputs += rebin.rebin.histout,


