# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import *

class dayabay_fastn_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.init_indices()
        if self.idx.ndim()!=1:
            raise self.exception('Expect 1d indexing')

        self.bindings=OrderedDict()

    def build(self):
        pass
        # bins = N.ascontiguousarray(self.cfg.bins, dtype='d')
        # emin, emax=self.cfg.normalize
        # try:
            # (imin, imax), = N.where( ((bins-emin)*(bins-emax)).round(6)==0.0 )
        # except:
            # raise Exception('Was able to determine normalization region for Fast Neutrons (%f, %f)'%(emin, emax))

        # self.integrator_gl = R.GaussLegendre(bins, self.cfg.order, bins.size-1, ns=self.common_namespace)
        # self.integrator_gl.points.setLabel('Fast n\nintegrationpoints')
        # self.objects['integrator'] = self.integrator_gl
        # with self.common_namespace:
            # for ns in self.namespaces:
                # fcn  = R.SelfPower(ns.name, ns=ns, bindings=self.bindings)
                # fcn.selfpower_inv.points( self.integrator_gl.points.x )
                # fcn.selfpower_inv.setLabel('Fast n shape:\n'+ns.name)

                # hist = R.GaussLegendreHist(self.integrator_gl, ns=ns)
                # hist.hist.f(fcn.selfpower_inv.result)
                # hist.hist.setLabel('Fast n hist:\n'+ns.name)

                # normalize = R.Normalize(imin, imax-imin, ns=ns)
                # normalize.normalize.inp( hist.hist.hist )
                # normalize.normalize.setLabel('Fast n hist:\n'+ns.name+' (norm)')

                # """Provide the outputs and objects"""
                # self.objects[('fcn', ns.name)]       = fcn
                # self.objects[('hist', ns.name)]      = hist
                # self.objects[('normalize', ns.name)] = normalize
                # self.transformations_out[ns.name]    = normalize.normalize
                # self.outputs[ns.name]                = normalize.normalize.out

    def define_variables(self):
        for it in self.idx.iterate():
            itname, = it.current_values()

            name = it.current_format('{name}{autoindex}', name=self.cfg.parameter)
            label = it.current_format('Fast neutron shape parameter for {site}')
            par = self.common_namespace.reqparameter(name, cfg=parcfg)
            par.setLabel(label)
