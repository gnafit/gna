
"""Oscillation probability Pee bundle v01.

Based on oscprob_v05. Updates:
    - Oscillation parameters were separated to a distinct bundle: oscpars_ee_v01
    - Keep only outputs
"""

from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import OrderedDict

class oscprob_ee_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 3, 'major')

        try:
            source_name, detector_name, component_name = self.cfg.bundle.major
        except:
            raise Exception('Unable to obtain major indices: source, detector and OP component')
        self.idx_source = self.nidx_major.get_subset(source_name)
        self.idx_detector = self.nidx_major.get_subset(detector_name)
        self.idx_component = self.nidx_major.get_subset(component_name)

    @staticmethod
    def _provides(cfg):
        return (), ('oscprob',)

    def build(self):
        self.comp0 = R.FillLike(1.0, labels='OP comp0')

        pmns_name = self.get_globalname('pmns')
        baseline_name = self.get_globalname('baseline')

        labelfmt = self.cfg.get('labelfmt', 'OP {component}:|{reactor}-\\>{detector}')

        modecos=True
        oscprobargs=[modecos]
        if 'dmnames' in self.cfg:
            oscprobargs.append(self.cfg['dmnames'])

        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name=baseline_name)

                oscprobkey = it_dist.current_format('{autoindex}')[1:]

                with self.namespace:
                    with self.namespace(pmns_name):
                        for it_minor in self.nidx_minor:
                            oscprob = self.context.objects[(pmns_name,oscprobkey)] = C.OscProb3(R.Neutrino.ae(), R.Neutrino.ae(), dist, *oscprobargs)

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
                                        trans.setLabel(it.current_format(labelfmt+'|'+it_minor.current_format()))
                                    else:
                                        trans.setLabel(it.current_format(labelfmt))
                                    output = trans[component]
                                    input  = trans['Enu']

                                self.set_input('oscprob',  it, input, argument_number=0)
                                self.set_output('oscprob', it, output)

    def define_variables(self):
        pass

