# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class detector_eres_common3(TransformationBundle):
    mode = 'correlated' # 'uncorrelated'
    def __init__(self, **kwargs):
        super(detector_eres_common3, self).__init__( **kwargs )
        self.transformations_in = self.transformations_out

    def build(self):
        with self.common_namespace:
            if self.mode=='correlated':
                eres = R.EnergyResolution(False, ns=self.common_namespace)
                for i, ns in enumerate(self.namespaces):
                    eres.add()

                    self.inputs[ns.name]  = eres.transformations[i].Nvis
                    self.outputs[ns.name] = eres.transformations[i].Nrec

                self.transformations_out[ns.name] = eres
            elif self.mode=='uncorrelated':
                for ns in self.namespaces:
                    eres = R.EnergyResolution(ns=ns)

                    self.transformations_out[ns.name] = eres

                    self.inputs[ns.name]  = eres.smear.Nvis
                    self.outputs[ns.name] = eres.smear.Nrec
            else:
                raise Exception( 'Invalid mode '+self.mode )

    def define_variables(self):
        for name, unc in self.cfg.pars.items():
            self.common_namespace.reqparameter(name, cfg=unc)

