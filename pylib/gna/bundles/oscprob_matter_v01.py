# -*- coding: utf-8 -*-
"""Oscillation probability (matter) bundle v01. Based on vacuum oscillation bundle oscprob_v04.

Changes since origin:
    - Switch from OscProb3 oscillation probability class to OscProbMatter
"""
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class oscprob_matter_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')

        try:
            source_name, detector_name = self.cfg.bundle.major
        except:
            raise Exception('Unable to obtain major indices: source, detector and OP component')

        self.idx_source = self.nidx_major.get_subset(source_name)
        self.idx_detector = self.nidx_major.get_subset(detector_name)

        if not 'density' in self.cfg:
            raise Exception('Density is not provided')

    @staticmethod
    def _provides(cfg):
        return ('pmns',), ('oscprob',)

    def build(self):
        pmns_name = self.get_globalname('pmns')
        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name='baseline')

                oscprobkey = it_dist.current_format('{autoindex}')[1:]

                with self.namespace, self.namespace(pmns_name):
                    for it_minor in self.nidx_minor:
                        oscprob = self.context.objects[(pmns_name, oscprobkey)] = C.OscProbMatter(R.Neutrino.ae(), R.Neutrino.ae(), dist)

                        it = it_source+it_detector

                        trans = oscprob.oscprob
                        label = it.current_format('OP:|{reactor}-\\>{detector}|')
                        if self.nidx_minor:
                            label+=it_minor.current_format()
                        trans.setLabel(label)

                        self.set_input('oscprob',  it, trans.Enu, argument_number=0)
                        self.set_output('oscprob', it, trans.oscprob)

    def define_variables(self):
        from gna.parameters.oscillation import reqparameters_reactor as reqparameters
        pmnspars_kwargs=dict()
        pdgyear = self.cfg.get('pdg_year', None)
        if pdgyear:
            pmnspars_kwargs['pdg_year']=pdgyear

        pmns_name = self.get_globalname('pmns')
        ns_pmns=self.namespace(pmns_name)
        reqparameters(ns_pmns, self.cfg.dm, **pmnspars_kwargs)

        names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
        with ns_pmns:
            R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), names, ns=ns_pmns)
            ns_pmns['Delta'].setFixed()
            ns_pmns['SigmaDecohRel'].setFixed()
            ns_pmns['SinSq23'].setFixed()
            ns_pmns.materializeexpressions()

        ns_pmns.reqparameter('rho', central=self.cfg.density, fixed=True, label='Matter density g/cm3')

        for i, vname in enumerate(names):
            ns_pmns[vname].setLabel('Psur(ee) weight %i: %s '%(i, vname))
