# -*- coding: utf-8 -*-

"""Parameters v04 bundle
Implements a set of unrelated parameters.

This version supports no indexes and not correlations.

Based on: parameters_v03
"""

from __future__ import print_function
from load import ROOT as R
from gna.bundle.bundle import *
import numpy as N
from gna import constructors as C

class parameters_v04(TransformationBundle):
    covmat, corrmat = None, None
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')
        self.check_nidx_dim(0, 0, 'minor')

    @staticmethod
    def _provides(cfg):
        names = tuple(cfg.pars.keys())
        return names, names if cfg.get('objectize') else ()

    def define_variables(self):
        self._par_container = []
        pars = self.cfg.pars
        labels = self.cfg.get('labels', {})
        objectize = self.cfg.get('objectize')

        for name, parcfg in pars.items():
            label = labels.get(name, '')
            par = self.reqparameter(name, None, cfg=parcfg, label=label)

            if objectize:
                trans=par.transformations.value
                trans.setLabel(label)
                self.set_output(name, it, trans.single())

                self._par_container.append(par)

    def build(self):
        pass
