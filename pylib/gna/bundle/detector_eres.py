# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from gna.bundle import *

@declare_bundle('eres_common3')
class detector_eres_common3(TransformationBundle):
    name = 'eres'
    parameters = [ 'Eres_a', 'Eres_b', 'Eres_c' ]
    def __init__(self, **kwargs):
        super(detector_eres_common3, self).__init__( **kwargs )

    def build(self):
        self.output=()
        for ns in self.namespaces:
            with ns:
                eres = R.EnergyResolution()
                self.output+=eres,

        return self.output

    def define_variables(self):
        for name, val, unc in zip( self.parameters, self.cfg.values, self.cfg.uncertainties ):
            self.common_namespace.reqparameter( name, central=val, uncertainty=unc, uncertainty_type=self.cfg.uncertainty_type )

