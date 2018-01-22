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

class bkg_weighted_hist_v01(TransformationBundle):
    name = 'bkg_weighted_hist'
    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.spectra = execute_bundle( cfg=self.cfg.spectra, storage=self.storage )
        self.namespaces = self.spectra.namespaces

        self.cfg.setdefault( 'name', self.cfg.parent_key() )
        print( 'Executing:\n', str(self.cfg), sep='' )

    def build(self):
        pass

    def define_variables(self):
        for ns in self.namespaces:
            if 'norm' in self.cfg:
                ns.reqparameter( '{}_norm'.format(self.cfg.name), self.cfg.norm )
            if 'rates' in self.cfg:
                ns.reqparameter( '{}_rate'.format(self.cfg.name), self.cfg.rates[ns.name] )
