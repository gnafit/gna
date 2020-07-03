# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class trans_weightedsumsq_v01(TransformationBundle):
    """Rebin bundle.
    Defines multiple rebin transformations.
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @staticmethod
    def _provides(cfg):
        return (), cfg.instances.keys()

    def build(self):
        self.objects = []
        for name, label in self.cfg.instances.items():
            if label is None:
                label = 'WeightedSumSq {autoindex}'

            for it in self.nidx_minor.iterate():
                rebin = C.Rebin(self.cfg.edges, self.cfg.rounding, labels=it.current_format(label))
                self.objects.append(rebin)

                self.set_input( name, it, rebin.rebin.histin, argument_number=0)
                self.set_output(name, it, rebin.rebin.histout)


