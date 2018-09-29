# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class efficiencies_v01(TransformationBundle):
    mode = 'correlated' # 'uncorrelated'
    def __init__(self, **kwargs):
        super(efficiencies_v01, self).__init__( **kwargs )
        self.transformations_in = self.transformations_out

    def define_variables(self):
        print('here')
        pass

