"""Oscillation probability in matter (approximate) bundle v02.

Based on oscprob_approx_v01, synchronized with oscrob_ee_v01

Updates since v01:
    - drop oscillation parameters (should be defined via oscpars_ee_v01)
    - add configuration validator
    - add option 'formula' to choose from 'juno-yb' or 'khan'
"""

from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from schema import Schema, Or, Optional, And, Use

class oscprob_approx_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

        try:
            source_name, detector_name = self.cfg.bundle.major
        except:
            raise Exception('Unable to obtain major indices: source, detector and OP component')
        self.idx_source = self.nidx_major.get_subset(source_name)
        self.idx_detector = self.nidx_major.get_subset(detector_name)

    _validator = Schema({
            'bundle': object,
            'formula': Or('juno-yb', 'juno yb', 'khan-approx'),
            Optional('density', default=2.6): Or(float, And((float,), lambda l: len(l)==2), None),
            Optional('electron_density'): Or(float, None)
        })

    @staticmethod
    def _provides(cfg):
        return (), ('oscprob_msw_approx',)

    def build(self):
        classes = {
                'juno yb': R.OscProb3ApproxMSW,
                'juno-yb': R.OscProb3ApproxMSW,
                'khan-approx': R.PsurEEMSWKhan
                }
        OpClass = classes[self.vcfg['formula']]

        pmns_name = self.get_globalname('pmns')
        baseline_name = self.get_globalname('baseline')
        electron_density = self.vcfg.get('electron_density')
        if electron_density and not self.vcfg['formula']=='khan-approx' :
            raise self.exception('May set electron density only for Khan et al.')
        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name=baseline_name)

                oscprobkey = it_dist.current_format('{autoindex}')[1:]

                with self.namespace, self.namespace(pmns_name):
                    for it_minor in self.nidx_minor:
                        if electron_density:
                            oscprob = OpClass(electron_density, dist)
                        else:
                            oscprob = OpClass(dist)

                        self.context.objects[(pmns_name,oscprobkey)] = oscprob

                        it = it_source+it_detector+it_minor

                        trans = oscprob.oscprob
                        if self.nidx_minor:
                            trans.setLabel( it.current_format('Approximate MSW OP: |{reactor}-\\>{detector}|'+it_minor.current_format()) )
                        else:
                            trans.setLabel( it.current_format('Approximate MSW OP: |{reactor}-\\>{detector}') )
                        output = trans.oscprob
                        enu_input  = trans['Enu']

                        self.set_input('oscprob_msw_approx',  it, enu_input, argument_number=0)
                        self.set_output('oscprob_msw_approx', it, output)

    def define_variables(self):
        pmns_name = self.get_globalname('pmns')
        ns_pmns=self.namespace(pmns_name)

        label_rho = 'Earth crust density in g/cm^3'
        density=self.vcfg['density']
        if isinstance(density, float):
            self.namespace.reqparameter("rho", central=density, fixed=True, label=label_rho)
        else:
            value, unc = density
            self.namespace.reqparameter("rho", central=value, sigma=unc, label=label_rho)

