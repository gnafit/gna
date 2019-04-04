# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *
from gna.expression import NIndex

class dummyvar_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

    @staticmethod
    def _provides(cfg):
        return tuple(cfg.variables.keys()), ()

    def define_variables(self):
        for name, var in self.cfg.variables.items():
            for i, nit in enumerate(self.nidx):
                self.reqparameter(name, nit, var)
