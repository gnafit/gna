# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from converters import convert
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import *

class dayabay_fastn_v01(TransformationBundle):
    name = 'dayabay_fastn'
    def __init__(self, **kwargs):
        super(dayabay_fastn_v01, self).__init__( **kwargs )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.pars.keys()]
        self.groups = Categories(self.cfg.get('groups', {}), recursive=True)
        self.bindings=OrderedDict()

    def build(self):
        from gna.parameters.printer import print_parameters
        print_parameters( self.common_namespace )

        bins = N.ascontiguousarray(self.cfg.bins, dtype='d')
        self.integrator_gl = R.GaussLegendre(bins, self.cfg.order, bins.size-1, ns=self.common_namespace)
        for ns in self.namespaces:
            fcn  = R.SelfPower(ns.name, ns=ns, bindings=self.bindings)
            fcn.selfpower_inv.points( self.integrator_gl.points.x )

            hist = R.GaussLegendreHist(self.integrator_gl, ns=ns)
            hist.hist.f(fcn.selfpower_inv.result)

            self.output_transformations+=hist,
            self.outputs+=hist.hist.data,

    def define_variables(self):
        for loc, unc in self.cfg.pars.items():
            name = self.groups.format_splitjoin(loc, self.cfg.formula)
            par = self.common_namespace.defparameter(name, cfg=unc)
            self.bindings[loc]=par
