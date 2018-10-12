# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import *

class parameters_1d_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.init_indices()
        if self.idx.ndim()!=1:
            raise self.exception('Expect 1d indexing')

    def define_variables(self):
        for it in self.idx.iterate():
            itname, = it.current_values()
            parcfg = self.cfg.pars[itname]

            name = it.current_format('{name}{autoindex}', name=self.cfg.parameter)
            label = it.current_format('Fast neutron shape parameter for {site}')
            par = self.common_namespace.reqparameter(name, cfg=parcfg)
            par.setLabel(label)
