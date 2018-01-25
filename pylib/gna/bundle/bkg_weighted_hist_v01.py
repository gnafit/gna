# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram
from gna.configurator import NestedDict, uncertain
from converters import convert

from gna.bundle import *

class bkg_weighted_hist_v01(TransformationBundle):
    name = 'bkg_weighted_hist'

    formats = dict(
            rate = '{}_rate',
            norm = '{}_norm',
            num  = '{}_num',
            livetime = 'livetime'
            )
    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.spectra = execute_bundle( cfg=self.cfg.spectra, storage=self.storage )
        self.namespaces = self.spectra.namespaces

        self.cfg.setdefault( 'name', self.cfg.parent_key() )
        print( 'Executing:\n', str(self.cfg), sep='' )

    def build(self):
        for ns in self.iterate_namespaces():
            # ws = R.WeightedSum()
            print(ns.name)

    def define_variables(self):
        pitems = None
        formula = [ self.formats[item].format( self.cfg.name ) for item in self.cfg.formula ]
        if len(formula)>1:
            pitems = convert( formula, 'stdvector' )

        self.products=[]
        numname = self.formats['num'].format( self.cfg.name )
        for ns in self.namespaces:
            for item in self.cfg.formula:
                num = self.cfg[item]

                if isinstance( num, uncertain ):
                    cnum = num
                else:
                    cnum = num[ns.name]
                ns.reqparameter( self.formats[item].format( self.cfg.name ), cnum )

            if pitems:
                with ns:
                    vp = R.VarProduct( pitems, numname, ns=ns )
                    ns[numname].get()
                    self.products.append( vp )
            else:
                ns.defparameter( numname, target=formula[0] )


