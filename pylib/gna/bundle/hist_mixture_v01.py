# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from converters import convert
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class hist_mixture_v01(TransformationBundle):
    name = 'hist_mixture'
    def __init__(self, **kwargs):
        super(hist_mixture_v01, self).__init__( **kwargs )

        if len(self.cfg.spectra)<2:
            raise Exception( 'hist_mixture_v01 should have at least 2 spectra defined' )

        self.spectra = OrderedDict([
                (name, execute_bundle(cfg=cfg, common_namespace=self.common_namespace, storage=self.storage))
                for name, cfg in self.cfg.spectra.items()
            ])

        for name, spectrum in self.spectra.items():
            if len(spectrum.outputs)!=1:
                raise Exception('hist_mixture_v01: expect only single output for each spectrum (exception for %s)'%name)

    def build(self):
        names = self.spectra.keys()
        for ns in self.namespaces:
            weights = [ ns.pathto('frac_'+name) for name in names ]

            ws = self.transformations[ns.name] = R.WeightedSum( convert(names, 'stdvector'), convert(weights, 'stdvector') )

            for name, spectrum in self.spectra.items():
                ws.sum.inputs[name]( spectrum.outputs[0] )

                self.outputs += ws.sum.sum,
                self.output_transformations+= ws,

    def define_variables(self):
        comb = '_'.join(('frac',)+tuple(sorted(self.cfg.spectra.keys()))+('comb',))

        self.vardiffs = OrderedDict()
        for ns in self.namespaces:
            ns.defparameter( name=comb, central=1, sigma=0.1, fixed=True )

            missing = self.cfg.spectra.keys()
            subst = [ns.pathto(comb)]
            for name, val in self.cfg.fractions.items():
                cname = 'frac_'+name
                ns.defparameter( cname, cfg=val )
                missing.pop(missing.index(name))
                subst.append(ns.pathto(cname))

            if len(missing)!=1:
                raise Exception('One weight of the hist_mixture should be autmatic')

            missing = 'frac_'+missing[0]
            vd = R.VarDiff( convert(subst, 'stdvector'), missing, ns=ns)
            ns[missing].get()
            self.vardiffs[ns.name] = vd
