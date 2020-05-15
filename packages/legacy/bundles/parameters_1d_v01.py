# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import gna.constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import *

class parameters_1d_v01(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        TransformationBundleLegacy.__init__(self, *args, **kwargs)
        self.init_indices()
        if self.idx.ndim()!=1:
            raise self.exception('Expect 1d indexing')

    def define_variables(self):
        sepunc = self.cfg.get('separate_uncertainty', False)
        for it in self.idx.iterate():
            itname, = it.current_values()
            parcfg = self.cfg.pars[itname]

            name = it.current_format(name=self.cfg.parameter)
            label = it.current_format(self.cfg.label)

            if parcfg.mode!='fixed' and sepunc:
                uncname = it.current_format(name=sepunc)
                unccfg = parcfg.get_unc()
                uncpar = self.common_namespace.reqparameter(uncname, cfg=unccfg)
                uncpar.setLabel(label+' (norm)')

                parcfg.mode='fixed'

            par = self.common_namespace.reqparameter(name, cfg=parcfg)
            par.setLabel(label)
