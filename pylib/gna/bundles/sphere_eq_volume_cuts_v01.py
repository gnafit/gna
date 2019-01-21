# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.configurator import NestedDict
from gna.constructors import stdvector
from gna.bundle import TransformationBundle

class sphere_eq_volume_cuts_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1)

    @staticmethod
    def _provides(cfg):
        return ('zone_weight', ), ()

    def define_variables(self):
        nzones = self.nidx.get_size()
        weight = 1.0/nzones

        name, = self.nidx.get_index_names()
        for i, it in enumerate(self.nidx):
            self.reqparameter('zone_weight', it, central=weight, fixed=True, label='Weight of zone {{{name}}}'.format(name=name) )
