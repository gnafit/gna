
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import gna.constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import TransformationBundle

class dayabay_fastn_power_v02(TransformationBundle):
    '''The implementation of Daya Bay fast neutron background using
    MC-inspired parametrization with power law.
    Diff from v01:
    - Usage of new IntegratorGL instead of legacy GaussLegendre
    '''
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1)

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        bins = self.context.outputs[self.cfg.bins].data()
        emin, emax=self.cfg.normalize
        try:
            (imin, imax), = N.where( ((bins-emin)*(bins-emax)).round(6)==0.0 )
        except:
            raise ValueError(f'Cannot determine normalization region for Fast Neutrons (emin, emax)')

        for i, it in enumerate(self.nidx.iterate()):
            parname = it.current_format(name=self.cfg.parameter)
            fcn  = R.SelfPower(parname, ns=self.namespace)
            integrator_gl = R.IntegratorGL(bins.size-1, self.cfg.order, bins, ns=self.namespace)
            integrator_gl.points.setLabel('Fast neutron energy points')
            fcn.selfpower_inv.points(integrator_gl.points.x)
            fcn.selfpower_inv.setLabel(it.current_format('Fast neutron shape {site}'))

            hist = integrator_gl.hist
            hist.setLabel(it.current_format('Fast neutron hist {site}'))

            fcn.selfpower_inv.result >> hist.f

            normalize = R.Normalize(int(imin), int(imax-imin))
            hist.outputs.back() >> normalize.normalize
            normalize.normalize.setLabel(it.current_format('Fast neutron hist {site} (norm)'))

            self.set_output(self.cfg.name, it, normalize.single())

            # """Provide the outputs and objects"""
            self.context.objects[('integrator',)+it.current_values()] = integrator_gl
            self.context.objects[('fcn',)+it.current_values()]       = fcn
            self.context.objects[('hist',)+it.current_values()]      = hist
            self.context.objects[('normalize',)+it.current_values()] = normalize

    def define_variables(self):
        pass
