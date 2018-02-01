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

class hist_mixture_v01(TransformationBundle):
    name = 'hist_mixture'
    def __init__(self, **kwargs):
        super(hist_mixture_v01, self).__init__( **kwargs )

    def build(self):
        file = R.TFile( self.cfg.filename, 'READ' )
        pass

    def define_variables(self):
        print( 'Hist mixture' )
        for ns in self.namespaces:
            print( 'here', ns.path, ns.name )

            for name, val in self.cfg.fractions.items():
                ns.defparameter( ns.pathto(name+'_frac'), cfg=val )

            from gna.parameters.printer import print_parameters
            print_parameters( ns )

            # import IPython
            # IPython.embed()

        from sys import exit
        exit(1)
