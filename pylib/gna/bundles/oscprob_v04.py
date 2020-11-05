"""Oscillation probability bundle v04. Updates since v03:
    - Switch from OscProbmPMNS to OscProb3 oscillation probability class

Miscellaneous:
    - Correct behavior for minor indices
    - Properly define the namespace name
"""

from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class oscprob_v04(TransformationBundle):
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

        assert 'dm' in self.cfg, "Expect 'dm' option"
        assert self.cfg.dm in ('23', 'ee'), "Expect 'dm' option to be '23', or 'ee'"

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
                            oscprob = self.context.objects[(pmns_name, oscprobkey)] = C.OscProb3(R.Neutrino.ae(), R.Neutrino.ae(), dist)

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

        for i, vname in enumerate(names):
            ns_pmns[vname].setLabel('Psur(ee) weight %i: %s '%(i, vname))
