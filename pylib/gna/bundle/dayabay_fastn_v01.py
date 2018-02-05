# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from converters import convert
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class dayabay_fastn_v01(TransformationBundle):
    name = 'dayabay_fastn'
    def __init__(self, **kwargs):
        super(dayabay_fastn_v01, self).__init__( **kwargs )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.variants]

    def build(self):
        pass

    def define_variables(self):
        for loc, unc in self.cfg.pars:
            print( loc, unc )

        from gna.parameters.printer import print_parameters
        print_parameters( env.common_namespace )

        import sys
        sys.exit(1)
