# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict

class dummy_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.objects=NestedDict()

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        for i, key in enumerate(self.nidx):
            self.make_trans(i, key)

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

        self.set_output(self.cfg.name, key, output)
        if input:
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    self.context.set_input(self.cfg.name, key, inp, argument_number=i)
            else:
                self.context.set_input(self.cfg.name, key, input, argument_number=0)

        self.objects[tkey] = obj
