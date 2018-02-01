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
from gna.grouping import Categories, CatDict

from gna.bundle import *

class bkg_weighted_hist_v01(TransformationBundle):
    name = 'bkg_weighted_hist'
    create_variable_links = True

    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.spectra = execute_bundle( cfg=self.cfg.spectra, storage=self.storage )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.variants]

        self.cfg.setdefault( 'name', self.cfg.parent_key() )
        self.numname = '{}_num'.format( self.cfg.name )

        self.groups = Categories( self.cfg.get('groups', {}) )
        print( 'Executing:\n', str(self.cfg), sep='' )

    def build(self):
        spectra = CatDict(self.groups, self.spectra.transformations)
        self.transformations=NestedDict()
        for ns in self.namespaces:
            labels  = convert([self.cfg.name], 'stdvector')
            weights = convert([ns.pathto(self.numname)], 'stdvector')
            ws = R.WeightedSum(labels, weights)

            inp = spectra[ns.name]
            ws.sum.inputs[self.cfg.name](inp.hist.hist)

            self.transformations(ns.name).hist = inp
            self.transformations[ns.name].sum = ws

            self.outputs += ws.sum.sum,
            self.output_transformations+=ws,

    def define_variables(self):
        self.products=[]

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

                item = self.groups.format_splitjoin( det, fullitem, prepend=self.common_namespace.path )
                formula.append(item)

                if self.create_variable_links and not head in ns:
                    self.common_namespace(det)[head] = item

            if len(formula)>1:
                vp = R.VarProduct(convert(formula, 'stdvector'), self.numname, ns=ns)
                ns[self.numname].get()
                self.products.append( vp )
            else:
                ns.defparameter( self.numname, target=formula[0] )
