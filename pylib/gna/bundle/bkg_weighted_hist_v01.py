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
from gna.grouping import GroupsSet

from gna.bundle import *

class bkg_weighted_hist_v01(TransformationBundle):
    name = 'bkg_weighted_hist'
    create_variable_links = True

    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.spectra = execute_bundle( cfg=self.cfg.spectra, storage=self.storage )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.variants]

        for det in self.cfg.parent(2).detectors:
            detns = self.common_namespace(det)
            detns.reqparameter('livetime', central=10, sigma=0.1, fixed=True)

        self.cfg.setdefault( 'name', self.cfg.parent_key() )
        self.numname = '{}_num'.format( self.cfg.name )
        print( 'Executing:\n', str(self.cfg), sep='' )

    def build(self):
        for ns in self.namespaces:
            labels  = convert((ns.pathto(self.cfg.name)), 'stdvector')
            weights = convert((ns.pathto(self.cfg.num_name)), 'stdvector')
            import IPython
            IPython.embed()
            # ws = R.WeightedSum()
            print(ns.name)

    def define_variables(self):
        self.products=[]

        groups = GroupsSet( self.cfg.get('groups', {}) )

        #
        # Define variables, which inputs are defined within the current config
        #
        for fullitem in self.cfg.formula:
            path, head = fullitem.rsplit('.', 1)
            numbers = self.cfg.get( head, {} )
            for loc, unc in numbers.items():
                self.common_namespace(loc).defparameter(head, cfg=unc)

        #
        # Link the other variables
        #
        for det in self.cfg.variants:
            ns = self.common_namespace(det)
            formula = []
            for fullitem in self.cfg.formula:
                path, head = fullitem.rsplit('.', 1)

                item = groups.format_splitjoin( det, fullitem, prepend=self.common_namespace.path )
                formula.append(item)

                if self.create_variable_links and not head in ns:
                    self.common_namespace(det)[head] = item

            if len(formula)>1:
                vp = R.VarProduct(convert(formula, 'stdvector'), self.numname, ns=ns)
                ns[self.numname].get()
                self.products.append( vp )
            else:
                ns.defparameter( self.numname, target=formula[0] )
