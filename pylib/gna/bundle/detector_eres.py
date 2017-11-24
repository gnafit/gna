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
        super(detector_eres_common3).__init__( **kwargs )

    def build(self):
        from file_reader import read_object_auto

        self.output=()
        for ns in self.namespaces:
            with ns:
                eres = R.EnergyResolution()
                self.output+=eres,

        return self.output

    def define_variables(self):
        for name, val, unc in zip( self.parameters, cfg.values, cfg.uncertainties ):
            env.defparameter( name, central=value, uncertainty=unc, uncertainty_type=cfg.uncertainty_type )

