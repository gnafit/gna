# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *
from gna.expression import NIndex

class dummyvar(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        super(dummyvar, self).__init__( *args, **kwargs )

    def define_variables(self):
        idx = self.cfg.indices
        if not isinstance(idx, NIndex):
            idx = NIndex(fromlist=self.cfg.indices)

        for name, var in self.cfg.variables.items():
            for i, nidx in enumerate(idx.iterate()):
                self.context.set_variable( name, nidx, var )
