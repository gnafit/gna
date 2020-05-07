# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import gna.constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import TransformationBundle

class histogram_flat_v01(TransformationBundle):
    """The bundle produces flat normalized histogram"""
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0,1)

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        edges = self.cfg.edges
        emin, emax=self.cfg.normalize
        try:
            (imin, imax), = N.where( ((edges-emin)*(edges-emax)).round(6)==0.0 )
        except:
            raise Exception('Was able to determine normalization region for Fast Neutrons (%f, %f)'%(emin, emax))

        data=N.zeros(edges.size-1)
        data[imin:imax]=1.0
        data/=data.sum()

        for it in self.nidx.iterate():
            hist = C.Histogram(edges, data)
            self.set_output(self.cfg.name, it, hist.single())

            # """Provide the outputs and objects"""
            self.context.objects[('hist',)+it.current_values()]      = hist

    def define_variables(self):
        pass
