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

    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.spectra = execute_bundle( cfg=self.cfg.spectra, common_namespace=self.common_namespace, storage=self.storage )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.variants]

        self.cfg.setdefault( 'name', self.cfg.parent_key() )

        self.groups = Categories( self.cfg.get('groups', {}), recursive=True )

    def build(self):
        self.transformations=NestedDict()
        spectra = CatDict(self.groups, self.spectra.transformations)

        targetfmt, formulafmt = self.get_target_formula()
        for ns in self.namespaces:
            target = self.groups.format_splitjoin(ns.name, targetfmt, prepend=self.common_namespace.path)

            labels  = convert([self.cfg.name], 'stdvector')
            weights = convert([target], 'stdvector')
            ws = R.WeightedSum(labels, weights)

            inp = spectra[ns.name]
            ws.sum.inputs[self.cfg.name](inp.single())

            self.transformations(ns.name).hist = inp
            self.transformations[ns.name].sum = ws

            self.outputs += ws.sum.sum,
            self.output_transformations+=ws,

    def define_variables(self):
        self.products=[]

        #
        # Define variables, which inputs are defined within the current config
        #
        for fullitem in self.iterate_formula():
            static = self.determine_variable( fullitem )
            numbers = self.cfg.get( static, {} )

            for loc, unc in numbers.items():
                path, head = self.groups.format_splitjoin( loc, fullitem ).rsplit( '.', 1 )
                self.common_namespace(path).defparameter(head, cfg=unc)

        #
        # Link the other variables
        #
        targetfmt, formulafmt = self.get_target_formula()
        for det in self.cfg.variants:
            ns = self.common_namespace(det)
            formula = []
            for fullitem in formulafmt:
                item = self.groups.format_splitjoin(det, fullitem, prepend=self.common_namespace.path)
                formula.append(item)

            target = self.groups.format_splitjoin(det, targetfmt)
            tpath, thead = target.rsplit('.', 1)
            tns = self.common_namespace(tpath)
            if len(formula)>1:
                vp = R.VarProduct(convert(formula, 'stdvector'), thead, ns=tns)

                tns[thead].get()
                self.products.append( vp )
            else:
                tns.defparameter( thead, target=formula[0] )

    def get_target_formula(self):
        formula = self.cfg.formula
        if isinstance(formula, basestring):
            target, formula = formula.split('=')
        else:
            target, formula = formula

        if isinstance(formula, basestring):
            formula = formula.split('*')

        return target, formula

    def iterate_formula(self):
        for item in self.get_target_formula()[1]:
            yield item

    def determine_variable(self, item):
        return '.'.join(i for i in item.split('.') if not ('{' in i or '}' in i))
