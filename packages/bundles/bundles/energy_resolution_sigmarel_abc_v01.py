from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

from gna.configurator import StripNestedDict
from tools.schema import *

class energy_resolution_sigmarel_abc_v01(TransformationBundle):
    """Detector energy resolution sigma parameterization v01

       Implements the energy resolution matrix, defined with the following relative sigma:

       # pars: sigma_e/e = sqrt(a^2 + b^2/E + c^2/E^2),
       # a - non-uniformity term
       # b - statistical term
       # c - dark noise term
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'both')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
            'bundle': object,
            'parameter': str,
            'pars': And((str,), haslength(exactly=3)),
            Optional('label', default= 'σ/E'): str,
        })

    @staticmethod
    def _provides(cfg):
        return (), ('eres_sigmarel', )

    def build(self):
        self.objects = []
        basename = self.vcfg['parameter']
        names = ['.'.join((basename, name)) for name in self.vcfg['pars']]

        sigma = C.EnergyResolutionSigmaRelABC(names, ns=self.namespace)
        self.objects.append(sigma)

        trans = sigma.sigma
        trans.setLabel(self.vcfg['label'])

        self.set_input('eres_sigmarel', None, trans.inputs.back(), argument_number=0)
        self.set_output('eres_sigmarel', None, trans.outputs.back())

    def define_variables(self):
        pass
