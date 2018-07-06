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

class hist_mixture_v01(TransformationBundle):
    def __init__(self, **kwargs):
        variants  = kwargs['cfg'].get('variants', None)
        if variants is not None:
            kwargs['namespaces'] = list(variants)
        super(hist_mixture_v01, self).__init__( **kwargs )

        if len(self.cfg.spectra)<2:
            raise Exception( 'hist_mixture_v01 should have at least 2 spectra defined' )

    def build(self):
        self.bundles = OrderedDict([
                (name, execute_bundle(cfg=cfg, namespaces=None, shared=self.shared)[0])
                for name, cfg in self.cfg.spectra.items()
            ])

        for name, spectrum in self.bundles.items():
            if len(spectrum.outputs)!=1:
                raise Exception('hist_mixture_v01: expect only single output for each spectrum (exception for %s)'%name)

        names = self.bundles.keys()
        for ns in self.namespaces:
            weights = [ ns.pathto('frac_'+name) for name in names ]

            ws = R.WeightedSum( stdvector(names), stdvector(weights), ns=ns )
            ws.sum.setLabel('Mixture\n'+ns.name)

            for name, spectrum in self.bundles.items():
                ws.sum.inputs[name]( spectrum.outputs.values()[0] )

            self.objects[('sum', ns.name)]    = ws
            self.transformations_out[ns.name] = ws.sum
            self.outputs[ns.name]             = ws.sum.sum

    def define_variables(self):
        for ns in self.namespaces:
            missing = self.cfg.spectra.keys()
            subst = []
            for name, val in self.cfg.fractions.items():
                cname = 'frac_'+name
                par = ns.reqparameter( cname, cfg=val )
                par.setLabel('{} fraction'.format(name))
                missing.pop(missing.index(name))
                subst.append(ns.pathto(cname))

            if len(missing)!=1:
                raise Exception('One weight of the hist_mixture should be autmatic')

            label='{} fraction: '.format(missing[0])
            label+='-'.join(['1']+['frac_'+n for n in self.cfg.fractions.keys()])

            missing = 'frac_'+missing[0]
            vd = R.VarDiff( stdvector(subst), missing, 1.0, ns=ns)
            par=ns[missing].get()
            self.objects[('vardiff', ns.name)] = vd
            par.setLabel(label)