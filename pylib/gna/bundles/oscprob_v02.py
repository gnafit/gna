from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class oscprob_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(3, 3, 'major')
        self.check_nidx_dim(0, 0, 'minor')

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

        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name='baseline')

                oscprobkey = it_dist.current_format('{autoindex}')[1:]
                with self.namespace:
                    with self.namespace('pmns'):
                        oscprob = self.context.objects[oscprobkey] = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae(), dist)

                for it_component in self.idx_component:
                    component, = it_component.current_values()
                    it = it_source+it_detector+it_component
                    if component=='comp0':
                        output = self.comp0.fill.outputs['a']
                        input = self.comp0.fill.inputs['a']
                    else:
                        if not component in oscprob.transformations:
                            raise Exception( 'No component %s in oscprob transformation'%component )

                        trans = oscprob.transformations[component]
                        trans.setLabel( it.current_format('OP {component}: {reactor}-\\>{detector}') )
                        output = trans[component]
                        input  = trans['Enu']

                    self.set_input('oscprob',  it, input, argument_number=0)
                    self.set_output('oscprob', it, output)

    def define_variables(self):
        from gna.parameters.oscillation import reqparameters
        ns_pmns=self.namespace('pmns')
        reqparameters(ns_pmns)

        names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
        with ns_pmns:
            R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), names, ns=ns_pmns)
            ns_pmns['Delta'].setFixed()
            ns_pmns['SigmaDecohRel'].setFixed()
            ns_pmns['SinSq23'].setFixed()
            ns_pmns.materializeexpressions()

        for i, vname in enumerate(names):
            ns_pmns[vname].setLabel('Psur(ee) weight %i: %s '%(i, vname))
