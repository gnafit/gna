# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class detector_eres_common3(TransformationBundle):
    name = 'eres'
    parameters = [ 'Eres_a', 'Eres_b', 'Eres_c' ]
    mode = 'correlated' # 'uncorrelated'
    def __init__(self, **kwargs):
        super(detector_eres_common3, self).__init__( **kwargs )

    def build(self):
        if self.mode=='correlated':
            eres = R.EnergyResolution( False )
            self.output_transformations+=eres,
            for i, ns in enumerate(self.namespaces):
                eres.add()

                self.inputs  += eres.transformations[i].Nvis,
                self.outputs += eres.transformations[i].Nrec,
        elif self.mode=='uncorrelated':
            for ns in self.namespaces:
                with ns:
                    eres = R.EnergyResolution()
                    self.output_transformations+=eres,

                    self.inputs  += eres.smear.Nvis,
                    self.outputs += eres.smear.Nrec,
        else:
            raise Exception( 'Invalid mode '+self.mode )

    def define_variables(self):
        for name, val, unc in zip( self.parameters, self.cfg.values, self.cfg.uncertainties ):
            self.common_namespace.reqparameter( name, central=val, uncertainty=unc, uncertainty_type=self.cfg.uncertainty_type )

