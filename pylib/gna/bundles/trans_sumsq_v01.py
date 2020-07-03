# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class trans_sumsq_v01(TransformationBundle):
    """SumSq bundle: defines SumSq object"""
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @staticmethod
    def _provides(cfg):
        return (), cfg.instances.keys()

    def build(self):
        self.objects = []
        ninputs = self.cfg['ninputs']
        for name, label in self.cfg.instances.items():
            if label is None:
                label = 'SumSq {autoindex}'

            for it in self.nidx_minor.iterate():
                sumsq = C.SumSq(labels=it.current_format(label))
                self.objects.append(sumsq)

                for i in range(ninputs):
                    inp = sumsq.add_input('input_{:02d}'.format(i))
                    self.set_input(name, it, inp, argument_number=i)

                self.set_output(name, it, sumsq.sumsq.sumsq)


