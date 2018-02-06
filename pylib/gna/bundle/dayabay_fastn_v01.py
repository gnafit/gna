# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from converters import convert
from gna.configurator import NestedDict
from gna.grouping import CatDict, Categories

from gna.bundle import *

class dayabay_fastn_v01(TransformationBundle):
    name = 'dayabay_fastn'
    def __init__(self, **kwargs):
        super(dayabay_fastn_v01, self).__init__( **kwargs )
        self.namespaces = [self.common_namespace(var) for var in self.cfg.pars.keys()]
        self.groups = self.cfg.get('groups', {})
        self.groups = Categories(self.groups, recursive=True)

    def build(self):
        pass

    def define_variables(self):
        for loc, unc in self.cfg.pars.items():
            name = self.groups.format_splitjoin( loc, self.cfg.formula)
            self.common_namespace.defparameter(name, cfg=unc)

        from gna.parameters.printer import print_parameters
        print_parameters( self.common_namespace )

        import sys
        sys.exit(1)
