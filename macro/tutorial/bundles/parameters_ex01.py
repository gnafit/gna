#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bundle.bundle import TransformationBundle

class parameters_ex01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0,0)

    def define_variables(self):
        for parname, parcfg in self.cfg.pars.items(nested=True):
            self.reqparameter(parname, None, cfg=parcfg)
