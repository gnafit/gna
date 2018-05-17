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

            self.transformations_in[tkey] = trans
            self.inputs[tkey] = trans.source
        else:
            obj = C.Points( N.zeros(shape=self.cfg.size, dtype='d') )
            trans = obj.points

        if self.cfg.debug:
            print( 'Create {var} [{inp}out]'.format(var=tkey, inp=self.cfg.input and 'in, ' or '') )

        if self.context:
            self.context.set_output(trans.single(), self.cfg.name, key, self.fmt)

        self.objects[tkey] = obj
        self.transformations_out[tkey] = trans
        self.shared[tkey] = self.outputs[tkey] = trans.single()
        trans.setLabel( tkey )

