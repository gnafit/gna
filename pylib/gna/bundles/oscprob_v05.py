# -*- coding: utf-8 -*-

"""Oscillation probability bundle v05.

Updates since v04:
    - Internal parameters configurations, not dependent to the global gna.parameters.oscillation
"""

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import OrderedDict

class oscprob_v05(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(3, 3, 'major')

        try:
            source_name, detector_name, component_name = self.cfg.bundle.major
        except:
            raise Exception('Unable to obtain major indices: source, detector and OP component')
        self.idx_source = self.nidx_major.get_subset(source_name)
        self.idx_detector = self.nidx_major.get_subset(detector_name)
        self.idx_component = self.nidx_major.get_subset(component_name)

    @staticmethod
    def _provides(cfg):
        return ('pmns',), ('oscprob',)

    def build(self):
        self.comp0 = R.FillLike(1.0, labels='OP comp0')

        pmns_name = self.get_globalname('pmns')
        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name='baseline')

                oscprobkey = it_dist.current_format('{autoindex}')[1:]

                with self.namespace:
                    with self.namespace(pmns_name):
                        for it_minor in self.nidx_minor:
                            oscprob = self.context.objects[(pmns_name,oscprobkey)] = C.OscProb3(R.Neutrino.ae(), R.Neutrino.ae(), dist)

                            for it_component in self.idx_component:
                                component, = it_component.current_values()
                                it = it_source+it_detector+it_component+it_minor
                                if component=='comp0':
                                    output = self.comp0.fill.outputs['a']
                                    input = self.comp0.fill.inputs['a']
                                else:
                                    if not component in oscprob.transformations:
                                        raise Exception( 'No component %s in oscprob transformation'%component )

                                    trans = oscprob.transformations[component]
                                    if self.nidx_minor:
                                        trans.setLabel( it.current_format('OP {component}:|{reactor}-\\>{detector}|'+it_minor.current_format()) )
                                    else:
                                        trans.setLabel( it.current_format('OP {component}:|{reactor}-\\>{detector}') )
                                    output = trans[component]
                                    input  = trans['Enu']

                                self.set_input('oscprob',  it, input, argument_number=0)
                                self.set_output('oscprob', it, output)

    def define_variables(self):
        pmnspars_kwargs=dict()
        pmns_name = self.get_globalname('pmns')
        ns_pmns=self.namespace(pmns_name)

        #
        # Define oscillation parameters
        #
        pars = self.cfg['parameters']
        otherpars = dict(
                SinSq23 = 0.542,
                )
        labels = OrderedDict([
            ('DeltaMSq23', 'Mass splitting |Δm²₂₃|'),
            ('DeltaMSq12', 'Mass splitting |Δm²₂₁|'),
            ('SinSqDouble13', 'Reactor mixing amplitude sin²2θ₁₃ '),
            ('SinSqDouble12', 'Solar mixing amplitude sin²2θ₁₂'),
            ('SinSq23', 'Atmospheric mixing angle sin²θ₂₃'),
            ])

        for name, label in labels.items():
            if name in pars:
                central, sigma = pars[name], None
            else:
                central, sigma = otherpars[name], None
            if isinstance(central, (tuple, list)):
                central, sigma=central

            if sigma:
                ns_pmns.reqparameter(name, central=central, sigma=sigma, label=label)
            else:
                ns_pmns.reqparameter(name, central=central, free=True, label=label)

        ns_pmns.reqparameter('Alpha', type='discrete', default='normal', variants={'normal': 1.0, 'inverted': -1.0}, label='Neutrino mass ordering α')
        ns_pmns.reqparameter('Delta', type='uniformangle', central=0.0, fixed=True, label='CP violation phase δ(CP)')

        #
        # Define oscillation expressions to provide missing and conjucated oscillation parameters
        # Define PMNS oscillation parameters
        #
        with ns_pmns:
            self._expressions_pars=C.OscillationExpressions(ns=ns_pmns)
            self._expressions_pmns=C.PMNSExpressionsC(ns=ns_pmns)

        #
        # Define oscillation weights
        #
        names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
        with ns_pmns:
            self._expressions_oscprob=R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), names, ns=ns_pmns)
            ns_pmns.materializeexpressions()

        for i, vname in enumerate(names):
            ns_pmns[vname].setLabel('Psur(ee) weight %i: %s '%(i, vname))

