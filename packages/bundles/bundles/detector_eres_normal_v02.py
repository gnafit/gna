from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

from gna.configurator import StripNestedDict
from tools.schema import *

class detector_eres_normal_v02(TransformationBundle):
    """Detector energy resolution parameterization v02

       Implements the energy resolution matrix, defined with the following relative sigma:

       # pars: sigma_e/e = sqrt(a^2 + b^2/E + c^2/E^2),
       # a - non-uniformity term
       # b - statistical term
       # c - dark noise term

    Updates since detector_eres_normal_v01:
    - Parameters should be loaded elsewhere.
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))
        self.labels = self.vcfg['labels']

    _validator = Schema({
            'bundle': object,
            'parameter': str,
            'pars': And((str,), haslength(exactly=3)),
            Optional('labels', default={}): {str: str},
            Optional('expose_matrix', default=False): bool,
            Optional('split_transformations', default=True): bool,
        })


    def set_label(self, obj, key, it, default=None, *args, **kwargs):
        if self.labels is False:
            return

        labelfmt = self.labels.get(key, default)
        if not labelfmt:
            return

        label = it.current_format(labelfmt, *args, **kwargs)
        obj.setLabel(label)

    @staticmethod
    def _provides(cfg):
        return (), ('eres_matrix', 'eres')

    def build(self):
        split_transformations = self.vcfg['split_transformations']

        self.objects = []
        expose_matrix = R.GNA.DataPropagation.Propagate if self.vcfg['expose_matrix'] else R.GNA.DataPropagation.Ignore
        for it_major in self.nidx_major:
            vals = it_major.current_values(name=self.vcfg['parameter'])
            names = ['.'.join(vals+(name,)) for name in self.vcfg['pars']]

            eres = C.EnergyResolution(names, expose_matrix, ns=self.namespace)
            self.objects.append(eres)

            self.set_label(eres.matrix, 'matrix', it_major, 'Energy resolution matrix ({autoindex})')
            self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)
            self.set_output('eres_matrix', it_major, eres.matrix.FakeMatrix)

            if not split_transformations:
                self.set_label(eres.smear, 'smear', it_major, 'Energy resolution ({autoindex})')

            trans = eres.smear
            for i, it_minor in enumerate(self.nidx_minor):
                it = it_major + it_minor
                if i:
                    if split_transformations:
                        trans = eres.add_transformation()
                eres.add_input()

                if split_transformations:
                    self.set_label(trans, 'smear', it, 'Energy resolution ({autoindex})')

                self.set_input('eres', it, trans.inputs.back(), argument_number=0)
                self.set_output('eres', it, trans.outputs.back())

    def define_variables(self):
        pass
