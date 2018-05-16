# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class dummy(TransformationBundle):
    def __init__(self, **kwargs):
        super(dummy, self).__init__( **kwargs )

    def build(self):
        from gna.expression import NIndex
        idx = NIndex(fromlist=self.cfg.indices)

        if self.cfg.input:
            for i, key in enumerate(idx.iterate( mode=self.cfg.format )):
                trans = R.Identity()
                trans.identity.setLabel( key )

                self.objects[key] = trans
                self.transformations_out[key] = trans.identity
                self.shared[key] = self.outputs[key] = trans.identity.target

                self.transformations_in[key] = trans.identity
                self.inputs[key] = trans.identity.source
        else:
            for i, key in enumerate(idx.iterate( mode=self.cfg.format )):
                trans = C.Points( N.zeros(shape=self.cfg.size, dtype='d') )
                trans.points.setLabel( key )

                self.objects[key] = trans
                self.transformations_out[key] = trans.points
                self.shared[key] = self.outputs[key] = trans.points.points




