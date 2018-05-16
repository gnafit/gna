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
        idx = self.cfg.indices
        if not isinstance(idx, NIndex):
            idx = NIndex(fromlist=self.cfg.indices)

        for i, key in enumerate(idx.iterate( mode=self.cfg.format )):
            self.make_trans( i, key )

    def make_trans(self, i, key):
        if self.cfg.input:
            obj = R.Identity()
            trans = obj.identity

            self.transformations_in[key] = trans
            self.inputs[key] = trans.source
        else:
            obj = C.Points( N.zeros(shape=self.cfg.size, dtype='d') )
            trans = obj.points

        if self.cfg.debug:
            print( 'Create {var} [{inp}out]'.format(var=key, inp=self.cfg.input and 'in, ' or '') )

        self.objects[key] = obj
        self.transformations_out[key] = trans
        self.shared[key] = self.outputs[key] = trans.single()
        trans.setLabel( key )

