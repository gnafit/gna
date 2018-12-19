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
        print(self.nidx.ndim())
        print(self.nidx.ident())
        import IPython
        IPython.embed()

        # sepunc = self.cfg.get('separate_uncertainty', False)
        # for it in self.idx.iterate():
            # itname, = it.current_values()
            # parcfg = self.cfg.pars[itname]

            # name = it.current_format(name=self.cfg.parameter)
            # label = it.current_format(self.cfg.label)

            # if parcfg.mode!='fixed' and sepunc:
                # uncname = it.current_format(name=sepunc)
                # unccfg = parcfg.get_unc()
                # uncpar = self.common_namespace.reqparameter(uncname, cfg=unccfg)
                # uncpar.setLabel(label+' (norm)')

                # parcfg.mode='fixed'

            # par = self.common_namespace.reqparameter(name, cfg=parcfg)
            # par.setLabel(label)
