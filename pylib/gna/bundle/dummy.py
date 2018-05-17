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

        self.fmt = self.cfg.get('format', '{name}{autoindex}')
        for i, key in enumerate(idx.iterate( mode='longitems', name=self.cfg.name)):
            self.make_trans( i, key )

    def make_trans(self, i, key):
        tkey = self.fmt.format(**dict(key))
        if self.cfg.input:
            obj = R.Identity()
            trans = obj.identity
            input = trans.source
            output = trans.target
        else:
            obj = C.Points( N.zeros(shape=self.cfg.size, dtype='d') )
            trans = obj.points
            output = trans.points
            input = None

        if self.cfg.debug:
            print( 'Create {var} [{inp}out]'.format(var=tkey, inp=self.cfg.input and 'in, ' or '') )

        if self.context:
            self.context.set_output(output, self.cfg.name, key, self.fmt)
            if input:
                self.context.set_input(input, self.cfg.name, key, self.fmt, clone=0)

        if input:
            self.transformations_in[tkey] = trans
            self.inputs[tkey] = input

        self.objects[tkey] = obj
        self.transformations_out[tkey] = trans
        self.shared[tkey] = self.outputs[tkey] = output
        trans.setLabel( tkey )

