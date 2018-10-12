# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.configurator import NestedDict
from constructors import stdvector

from gna.bundle import *
from gna.bundle.connections import pairwise

class var_fractions_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.init_indices()
        if self.idx.ndim()!=0:
            raise self.exception('Does not support indexing, got {!s}'.format(self.idx))

    def define_variables(self):
        names_all = set(self.cfg.names)
        names_unc = self.cfg.fractions.keys()
        names_eval = names_all-set(names_unc)
        if len(names_eval)!=1:
            raise self.exception('User should provide N-1 fractions, the last one is not independent\n'
                                 'all: {!s}\nfractions: {!s}'.format(self.cfg.names, names_unc))
        name_eval=names_eval.pop()

        subst = []
        names = ()
        for name, val in self.cfg.fractions.items():
            cname = self.cfg.format.format(component=name)
            names+=cname,
            par = self.common_namespace.reqparameter( cname, cfg=val )
            par.setLabel('{} fraction'.format(name))
            subst.append(self.common_namespace.pathto(cname))

        label='{} fraction: '.format(name_eval)
        label+='-'.join(('1',)+names)

        name_eval = self.cfg.format.format(component=name_eval)
        with self.common_namespace:
            self.vd = R.VarDiff( stdvector(subst), name_eval, 1.0, ns=self.common_namespace)
            par=self.common_namespace[name_eval].get()

        par.setLabel(label)
