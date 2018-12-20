# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_eres_common3(TransformationBundleLegacy):
    mode = 'correlated' # 'uncorrelated'
    def __init__(self, *args, **kwargs):
        super(detector_eres_common3, self).__init__( *args, **kwargs )
        self.transformations_in = self.transformations_out

    def build(self):
        with self.common_namespace:
            if self.mode=='correlated':
                eres = R.EnergyResolutionC(False, ns=self.common_namespace)
                for i, ns in enumerate(self.namespaces):
                    eres.add()
                    eres.transformations[i].setLabel('Energy resolution:\n'+ns.name)

                    """Save transformations"""
                    self.transformations_out[ns.name] = eres.transformations[i]
                    self.inputs[ns.name]              = eres.transformations[i].Ntrue
                    self.outputs[ns.name]             = eres.transformations[i].Nrec

                    """Define observables"""
                    self.addcfgobservable(ns, eres.transformations[i].Nrec, 'eres', ignorecheck=True)

                self.objects['eres'] = eres
            elif self.mode=='uncorrelated':
                for ns in self.namespaces:
                    eres = R.EnergyResolutionC(ns=ns)
                    eres.smear.setLabel('Energy resolution')

                    """Save transformations"""
                    self.objects[('eres', ns.name)]   = eres
                    self.transformations_out[ns.name] = eres.smear
                    self.inputs[ns.name]              = eres.smear.Ntrue
                    self.outputs[ns.name]             = eres.smear.Nrec

                    """Define observables"""
                    self.addcfgobservable(ns, eres.smear.Nrec, 'eres', ignorecheck=True)
            else:
                raise Exception( 'Invalid mode '+self.mode )

    def define_variables(self):
        descriptions=[
                'spatial/temporal resolution',
                'photon statistics',
                'dark noise'
                ]
        for i, (name, unc) in enumerate(self.cfg.pars.items()):
            par = self.common_namespace.reqparameter(name, cfg=unc)
            par.setLabel( 'Energy resolution ({})'.format(descriptions[i]) )

