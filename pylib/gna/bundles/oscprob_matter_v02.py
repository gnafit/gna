"""Oscillation probability (matter) bundle v02.

Based on matter oscillation bundle oscprob_matter_v01. Synchronized with oscprob_ee_v01

Changes since oscprob_matter_v01:
    - Do not define oscillation variables, it should be done with oscpars_ee
"""

from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class oscprob_matter_v02(TransformationBundle):
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
        return (), ('oscprob',)

    def build(self):
        pmns_name = self.get_globalname('pmns')
        baseline_name = self.get_globalname('baseline')
        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name=baseline_name)

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
        self.namespace.reqparameter('rho', central=self.cfg.density, fixed=True, label='Matter density g/cm3')
