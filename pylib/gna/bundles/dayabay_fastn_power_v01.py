# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import gna.constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import TransformationBundle

class dayabay_fastn_power_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1,1)

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        bins = self.context.outputs[self.cfg.bins].data()
        emin, emax=self.cfg.normalize
        try:
            (imin, imax), = N.where( ((bins-emin)*(bins-emax)).round(6)==0.0 )
        except:
            raise Exception('Was able to determine normalization region for Fast Neutrons (%f, %f)'%(emin, emax))

        self.integrator_gl = R.GaussLegendre(bins, self.cfg.order, bins.size-1, ns=self.namespace)
        self.integrator_gl.points.setLabel('Fast neutron\nintegration points')
        for it in self.nidx.iterate():
            parname = it.current_format(name=self.cfg.parameter)
            fcn  = R.SelfPower(parname, ns=self.namespace)
            fcn.selfpower_inv.points(self.integrator_gl.points.x)
            fcn.selfpower_inv.setLabel(it.current_format('Fast neutron shape\n{site}'))

            hist = R.GaussLegendreHist(self.integrator_gl)
            hist.hist.f(fcn.selfpower_inv.result)
            hist.hist.setLabel(it.current_format('Fast neutron hist\n{site}'))

            normalize = R.Normalize(imin, imax-imin)
            normalize.normalize.inp( hist.hist.hist )
            normalize.normalize.setLabel(it.current_format('Fast neutron hist\n{site} (norm)'))

            self.set_output(self.cfg.name, it, normalize.single())

            # """Provide the outputs and objects"""
            self.context.objects[('fcn',)+it.current_values()]       = fcn
            self.context.objects[('hist',)+it.current_values()]      = hist
            self.context.objects[('normalize',)+it.current_values()] = normalize

    def define_variables(self):
        pass
