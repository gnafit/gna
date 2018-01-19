# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class bkg_weighted_hist_v01(TransformationBundle):
    name = 'bkg_weighted_hist'
    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

    def build(self):
        pass

    def define_variables(self):
        pass
