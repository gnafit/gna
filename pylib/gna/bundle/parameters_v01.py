# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
import gna.constructors as C
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle.bundle import *

class parameters_v01(TransformationBundleV01):
    def __init__(self, *args, **kwargs):
        TransformationBundleV01.__init__(self, *args, **kwargs)

    def define_variables(self):
        sepunc = self.cfg.get('separate_uncertainty', False)
        parname = self.cfg.parameter
        pars = self.cfg.pars
        labelfmt = self.cfg.get('label', '')
        for it in self.nidx:
            nidx_values = it.current_values()
            parcfg = pars[nidx_values]
            label = it.current_format(labelfmt) if labelfmt else ''

            # if parcfg.mode!='fixed' and sepunc:
                # uncname = it.current_format(name=sepunc)
                # unccfg = parcfg.get_unc()
                # uncpar = self.common_namespace.reqparameter(uncname, cfg=unccfg)
                # uncpar.setLabel(label+' (norm)')

                # parcfg.mode='fixed'

            par = self.reqparameter(parname, it, cfg=parcfg, label=label)
