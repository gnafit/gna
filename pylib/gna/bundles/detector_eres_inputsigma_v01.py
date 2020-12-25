
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_eres_inputsigma_v01(TransformationBundle):
    """Detector energy resolution bundle
    Implements gaussian energy resolution with sigma(E) read from an input file

    Changes since detector_eres_normal_v01:
        - Input sigma from the file
        - Parameters are deprecated

    Configuration options:
        - filename - input filename

    Optional configuration options:
        - expose_matrix=False        - if not set or False, the matrix will be propagated internally (output will be empty)
        - split_transformations=True - if True use distinct transformation for each input.
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.objects = []

        self.labels = self.cfg.get('labels', {})

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
        return (), ('eres_matrix', 'eres', 'eres_sigma')

    def read_data(self):
        from tools.data_load import read_object_auto
        self.input = read_object_auto(self.cfg.filename, convertto='array')

    def build(self):
        self.read_data()

        expose_matrix = self.cfg.get('expose_matrix', False)
        split_transformations = self.cfg.get('split_transformations', True)

        sigmas = C.Points(self.input[1])
        self.objects.append(sigmas)
        self.set_output('eres_sigma', None, sigmas.points.points)

        self.objects = []
        for it_major in self.nidx_major:
            eres = C.EnergyResolutionInput(expose_matrix)
            self.objects.append(eres)
            sigmas.points.points >> eres.matrix.RelSigma

            self.set_label(eres.matrix, 'matrix', it_major, 'Energy resolution matrix ({autoindex})')
            self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)
            self.set_input('eres_matrix', it_major, eres.matrix.RelSigma, argument_number=1)
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
