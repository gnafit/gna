# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class dummy(TransformationBundle):
    def __init__(self, **kwargs):
        super(dummy, self).__init__( **kwargs )

    def build(self):
        from gna.expression import NIndex
        idx = self.cfg.indices
        if not isinstance(idx, NIndex):
            idx = NIndex(fromlist=self.cfg.indices)

        for i, key in enumerate(idx.iterate()):
            self.make_trans( i, key )

    def make_trans(self, i, key):
        tkey = key.current_format(name=self.cfg.name)

        obj = R.Dummy(self.cfg.size, tkey)
        trans = obj.dummy

        if 'label' in self.cfg:
            trans.setLabel(key.current_format(self.cfg.label, name=self.cfg.name))

        output = obj.add_output('output')
        if self.cfg.get('input', False):
            ninputs = int(self.cfg.input)
            if ninputs>1:
                input = tuple(obj.add_input('input_%02d'%i) for i in range(self.cfg.input))
            else:
                input = obj.add_input('input')
        else:
            input = None

        if self.cfg.debug:
            print( 'Create {var} [{inp}out]'.format(var=tkey, inp=self.cfg.input and 'in, ' or '') )

        if self.context:
            self.context.set_output(output, self.cfg.name, key)
            if input:
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        self.context.set_input(inp, self.cfg.name, key, clone=i)
                else:
                    self.context.set_input(input, self.cfg.name, key, clone=0)

        if input:
            self.transformations_in[tkey] = trans

            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    self.inputs[(tkey, '%02i'%i)] = inp
            else:
                self.inputs[tkey] = input

        self.objects[tkey] = obj
        self.transformations_out[tkey] = trans
        self.shared[tkey] = self.outputs[tkey] = output
