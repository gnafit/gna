from load import ROOT as R
import numpy as N
from gna.env import env, namespace
import gna.constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import *

class dayabay_fastn_v02(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        TransformationBundleLegacy.__init__(self, *args, **kwargs)
        self.init_indices()
        if self.idx.ndim()!=1:
            raise self.exception('Expect 1d indexing')

        self.bindings=dict()

    def build(self):
        bins = self.context.outputs[self.cfg.bins].data()
        emin, emax=self.cfg.normalize
        try:
            (imin, imax), = N.where( ((bins-emin)*(bins-emax)).round(6)==0.0 )
        except:
            raise Exception('Was able to determine normalization region for Fast Neutrons (%f, %f)'%(emin, emax))

        self.integrator_gl = R.GaussLegendre(bins, self.cfg.order, bins.size-1, ns=self.common_namespace)
        self.integrator_gl.points.setLabel('Fast neutron\nintegration points')
        with self.common_namespace:
            for it in self.idx.iterate():
                parname = it.current_format(name=self.cfg.parameter)
                fcn  = R.SelfPower(parname, ns=self.common_namespace)
                fcn.selfpower_inv.points(self.integrator_gl.points.x)
                fcn.selfpower_inv.setLabel(it.current_format('Fast neutron shape\n{site}'))

                hist = R.GaussLegendreHist(self.integrator_gl)
                hist.hist.f(fcn.selfpower_inv.result)
                hist.hist.setLabel(it.current_format('Fast neutron hist\n{site}'))

                normalize = R.Normalize(imin, imax-imin)
                normalize.normalize.inp( hist.hist.hist )
                normalize.normalize.setLabel(it.current_format('Fast neutron hist\n{site} (norm)'))

                self.set_output(normalize.single(), self.cfg.name, it)

                # """Provide the outputs and objects"""
                self.objects[('fcn',)+it.current_values()]       = fcn
                self.objects[('hist',)+it.current_values()]      = hist
                self.objects[('normalize',)+it.current_values()] = normalize

    def define_variables(self):
        pass
