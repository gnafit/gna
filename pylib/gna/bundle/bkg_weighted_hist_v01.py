# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram, stdvector
from gna.configurator import NestedDict, uncertain
from gna.grouping import Categories, CatDict

from gna.bundle import *

class bkg_weighted_hist_v01(TransformationBundle):
    def __init__(self, **kwargs):
        variants  = kwargs['cfg'].get('variants', None)
        if variants is not None:
            kwargs['namespaces'] = list(variants)
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.bundles, = execute_bundle(cfg=self.cfg.spectra, shared=self.shared)

        try:
            self.cfg.setdefault( 'name', self.cfg.parent_key() )
        except KeyError:
            if not 'name' in self.cfg:
                raise Exception( 'Background name is not specified' )

        self.groups = Categories( self.cfg.get('groups', {}), recursive=True )

    def build(self):
        spectra = CatDict(self.groups, self.bundles.outputs)

        targetfmt, formulafmt = self.get_target_formula()
        for ns in self.namespaces:
            target = self.groups.format_splitjoin(ns.name, targetfmt, prepend=self.common_namespace.path)

            labels  = stdvector([self.cfg.name])
            weights = stdvector([target])
            ws = R.WeightedSum(labels, weights, ns=ns)

            inp = spectra[ns.name]
            ws.sum.inputs[self.cfg.name](inp)

            """Save transformations"""
            self.objects[('spec',ns.name)]    = inp
            self.objects[('sum', ns.name)]    = ws
            self.transformations_out[ns.name] = ws.sum
            self.outputs[ns.name]             = ws.sum.sum

            """Add observables"""
            self.addcfgobservable(ns, ws.sum.sum, 'bkg/{name}', fmtdict=dict(name=self.cfg.name))

    def define_variables(self):
        #
        # Define variables, which inputs are defined within the current config
        #
        for fullitem in self.iterate_formula():
            static = self.determine_variable( fullitem )
            numbers = self.cfg.get( static, {} )

            for loc, unc in numbers.items():
                path, head = self.groups.format_splitjoin( loc, fullitem ).rsplit( '.', 1 )
                self.common_namespace(path).reqparameter(head, cfg=unc)

        #
        # Link the other variables
        #
        targetfmt, formulafmt = self.get_target_formula()
        for ns in self.namespaces:
            variant = ns.name
            formula = []
            for fullitem in formulafmt:
                item = self.groups.format_splitjoin(variant, fullitem, prepend=self.common_namespace.path)
                formula.append(item)

            target = self.groups.format_splitjoin(variant, targetfmt)
            tpath, thead = target.rsplit('.', 1)
            tns = self.common_namespace(tpath)
            if len(formula)>1:
                vp = R.VarProduct(stdvector(formula), thead, ns=tns)

                tns[thead].get()
                self.objects[('prod', variant)]=vp
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
