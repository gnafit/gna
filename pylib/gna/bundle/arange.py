# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class arange(TransformationBundle):
    def __init__(self, **kwargs):
        super(arange, self).__init__( **kwargs )

    def build(self):
        from gna.expression import NIndex
        idx = self.cfg.indices
        if not isinstance(idx, NIndex):
            idx = NIndex(fromlist=self.cfg.indices)

        self.data = N.arange(*self.cfg.args, dtype='d')
        for i, key in enumerate(idx.iterate()):
            self.make_trans( i, key )

    def make_trans(self, i, key):
        tkey = key.current_format('{name}{autoindex}', name=self.cfg.name)

        obj = C.Points(self.data)
        trans = obj.points
        trans.setLabel(tkey)
        output = trans.points

        self.set_output(output, self.cfg.name, key)

        self.objects[tkey] = obj
        self.transformations_out[tkey] = trans
        self.outputs[tkey] = output
