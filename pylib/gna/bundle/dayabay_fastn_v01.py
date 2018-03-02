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
    def __init__(self, **kwargs):
        super(dayabay_fastn_v01, self).__init__( **kwargs )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.pars.keys()]
        self.groups = Categories(self.cfg.get('groups', {}), recursive=True)
        self.bindings=OrderedDict()

    def build(self):
        bins = N.ascontiguousarray(self.cfg.bins, dtype='d')
        emin, emax=self.cfg.normalize
        try:
            (imin, imax), = N.where( ((bins-emin)*(bins-emax)).round(6)==0.0 )
        except:
            raise Exception('Was able to determine normalization region for Fast Neutrons (%f, %f)'%(emin, emax))

        self.integrator_gl = R.GaussLegendre(bins, self.cfg.order, bins.size-1, ns=self.common_namespace)
        self.integrator_gl.points.setLabel('Integration\npoints')
        self.objects['integrator'] = self.integrator_gl
        with self.common_namespace:
            for ns in self.namespaces:
                fcn  = R.SelfPower(ns.name, ns=ns, bindings=self.bindings)
                fcn.selfpower_inv.points( self.integrator_gl.points.x )
                fcn.selfpower_inv.setLabel('Fast n shape:\n'+ns.name)

                hist = R.GaussLegendreHist(self.integrator_gl, ns=ns)
                hist.hist.f(fcn.selfpower_inv.result)
                hist.hist.setLabel('Fast n hist:\n'+ns.name)

                normalize = R.Normalize(imin, imax-imin, ns=ns)
                normalize.normalize.inp( hist.hist.hist )
                normalize.normalize.setLabel('Fast n hist:\n'+ns.name+' (norm)')

                """Provide the outputs and objects"""
                self.objects[('fcn', ns.name)]       = fcn
                self.objects[('hist', ns.name)]      = hist
                self.objects[('normalize', ns.name)] = normalize
                self.transformations_out[ns.name]    = normalize.normalize
                self.outputs[ns.name]                = normalize.normalize.out

    def define_variables(self):
        for loc, unc in self.cfg.pars.items():
            name = self.groups.format_splitjoin(loc, self.cfg.formula)
            par = self.common_namespace.reqparameter(name, cfg=unc)
            self.bindings[loc]=par
            par.setLabel('Fast neutron shape parameter for '+loc)
