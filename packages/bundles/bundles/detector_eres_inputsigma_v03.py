from load import ROOT as R
import gna.constructors as C
from gna.configurator import StripNestedDict
from gna.bundle import TransformationBundle
from tools.schema import Schema, Optional, Or
from typing import Dict, Any

class detector_eres_inputsigma_v03(TransformationBundle):
    """Detector energy resolution bundle
    Implements gaussian energy resolution with sigma(E), passed as an input

    Defines:
    - Inputs:
        + eres - histogram for smearing
        + eres_sigmarel_input - input for the sigma, computed at the bin centers
    - Outputs:
        + eres - smeared histogram
        + eres_matrix - smearing matrix

    Changes since detector_eres_inputsigma_v02:
        - Rename 'mode' to 'eres_mode'
        - Add an option to use EnergyResolutionErfInputs: eres_mode='erf'
        - Add an option 'dynamic_edges' to enable dynamic_edges for EnergyResolutionErfInputs

    Optional configuration options:
        - eres_mode='erf', 'erf-square' or 'bincenter' - specifies the approximation to use:
            + 'erf' (default) - analytic integrals, input and output edges are the same (square matrix)
            + 'erf-square' - analytic integrals, input and output edges are different (matrix may be non square)
            + 'bincenter' - normal distribution at bin centers, input and output edges are the same (square matrix)
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
            Optional('eres_mode', default='erf'): Or('erf', 'erf-square', 'bincenter', 'bincenter-square'),
            Optional('expose_matrix', default=False): bool,
            Optional('split_transformations', default=True): bool,
            Optional('dynamic_edges', default=False): bool
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

        eres_mode = self.vcfg['eres_mode']
        edges_mode = R.GNA.DataMutability.Dynamic if self.vcfg['dynamic_edges'] else R.GNA.DataMutability.Static
        if eres_mode=='erf':
            def makeeres(it_major):
                eres = C.EnergyResolutionErfInputs(edges_mode)
                self.set_input('eres_matrix', it_major, eres.matrix.HistEdgesOut, argument_number=0)
                self.set_input('eres_matrix', it_major, eres.matrix.EdgesIn, argument_number=1)

                return eres
        elif eres_mode=='erf-square':
            def makeeres(it_major):
                eres = C.EnergyResolutionErfInput()
                self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)

                return eres
        elif eres_mode=='bincenter':
            def makeeres(it_major):
                eres = C.EnergyResolutionInputs(edges_mode)
                self.set_input('eres_matrix', it_major, eres.matrix.HistEdgesOut, argument_number=0)
                self.set_input('eres_matrix', it_major, eres.matrix.EdgesIn, argument_number=1)

                return eres
        elif eres_mode=='bincenter-square':
            def makeeres(it_major):
                eres = C.EnergyResolutionInput(expose_matrix)
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
