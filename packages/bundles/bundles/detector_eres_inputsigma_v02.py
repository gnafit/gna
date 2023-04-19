from load import ROOT as R
import gna.constructors as C
from gna.configurator import StripNestedDict
from gna.bundle import TransformationBundle
from tools.schema import Schema, Optional, Or
from typing import Dict, Any

class detector_eres_inputsigma_v02(TransformationBundle):
    """Detector energy resolution bundle
    Implements gaussian energy resolution with sigma(E), passed as an input

    Defines:
    - Inputs:
        + eres - histogram for smearing
        + eres_sigmarel_input - input for the sigma, computed at the bin centers
    - Outputs:
        + eres - smeared histogram
        + eres_matrix - smearing matrix

    Changes since detector_eres_inputsigma_v01:
        - Declare an input for the sigma
        - Deprecate file reading
        - Enable configuration validation

    Optional configuration options:
        - mode='bincenter' or 'erf'  - specifies the approximation: use normal distribution in bin centers (default) or use analytic integrals
        - expose_matrix=False        - if not set or False, the matrix will be propagated internally (output will be empty). Note: mode 'erf' always exposes the matrix.
        - split_transformations=True - if True use distinct transformation for each input.

    TODO for the future versions:
        - Set default mode to 'erf'
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.objects = []

        self.vcfg: Dict[str, Any] = self._validator.validate(StripNestedDict(self.cfg))
        self.labels = self.vcfg['labels']

    _validator = Schema({
            'bundle': object,
            Optional('labels', default={}): {str: str},
            Optional('mode', default='bincenter'): Or('bincenter', 'erf'),
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
    def _provides(_):
        return (), ('eres_matrix', 'eres', 'eres_sigmarel_input')

    def build(self):
        expose_matrix = R.GNA.DataPropagation.Propagate if self.vcfg['expose_matrix'] else R.GNA.DataPropagation.Ignore
        split_transformations = self.vcfg['split_transformations']

        eres_mode = self.vcfg['mode']
        if eres_mode=='bincenter':
            def makeeres(it_major):
                eres = C.EnergyResolutionInput(expose_matrix)
                self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)

                return eres
        elif eres_mode=='erf':
            def makeeres(it_major):
                eres = C.EnergyResolutionErfInput()
                self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)

                return eres
        else:
            assert False

        self.objects = []
        for it_major in self.nidx_major:
            eres = makeeres(it_major)
            self.objects.append(eres)
            self.set_input('eres_sigmarel_input', it_major, eres.matrix.RelSigma, argument_number=0)

            self.set_label(eres.matrix, 'matrix', it_major, 'Energy resolution matrix ({autoindex})')
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
