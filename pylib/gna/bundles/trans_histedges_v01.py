# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class trans_histedges_v01(TransformationBundle):
    """HistEdges bundle: defines HistEdges object with edges, bin centers and bin widths"""
    types = set(('edges', 'centers', 'widths'))
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @classmethod
    def _provides(cls, cfg):
        types = cfg.get('types')
        types = cls.types if types is None else cls.types.intersection(types)
        names = cfg.instances.keys()
        outputs = ['{}_{}'.format(name, t) for name in names for t in types]
        return (), outputs

    def build(self):
        self.objects = []
        types = self.cfg.get('types')
        types = self.types if types is None else self.types.intersection(types)
        for name, label in self.cfg.instances.items():
            if label is None:
                label = 'HistEdges {autoindex}'

            for it in self.nidx_minor.iterate():
                histedges = C.HistEdges(labels=it.current_format(label))
                self.objects.append(histedges)
                trans = histedges.histedges

                for t in types:
                    cname = '{}_{}'.format(name, t)
                    self.set_input(cname, it, trans.hist, argument_number=0)
                    self.set_output(cname, it, trans.outputs[t])


