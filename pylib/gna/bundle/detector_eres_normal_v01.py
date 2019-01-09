# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_eres_normal_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

    @staticmethod
    def _provides(cfg):
        return (cfg.parameter), ('eres_matrix', 'eres')

    def build(self):
        self.objects = []
        for it_major in self.nidx_major:
            vals = it_major.current_values(name=self.cfg.parameter)
            names = [ '.'.join(vals+(name,)) for name in self.names ]

            eres = C.EnergyResolution(names, ns=self.namespace)
            self.objects.append(eres)

            self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)

            trans = eres.smear
            for i, it_minor in enumerate(self.nidx_minor):
                it = it_major + it_minor
                eres.add_input()

                self.set_input('eres', it, trans.inputs.back(), argument_number=0)
                self.set_output('eres', it, trans.outputs.back())

    def define_variables(self):
        parname = self.cfg.parameter
        parscfg = self.cfg.pars
        self.names = None

        for it_major in self.nidx_major:
            major_values = it_major.current_values()
            pars = parscfg[major_values]

            if self.names is None:
                self.names = tuple(pars.keys())
            else:
                assert self.names == tuple(pars.keys())

            for i, (name, unc) in enumerate(pars.items()):
                it=it_major

                par = self.reqparameter(parname, it, cfg=unc, extra=name)


