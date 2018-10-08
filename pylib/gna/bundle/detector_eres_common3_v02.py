# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class detector_eres_common3_v02(TransformationBundle):
    common_matrix=True
    def __init__(self, **kwargs):
        super(detector_eres_common3_v02, self).__init__( **kwargs )
        self.transformations_in = self.transformations_out

        self.init_indices()

    def build(self):
        with self.common_namespace:
            eres = self.eres = R.EnergyResolution(False, ns=self.common_namespace)

            eres.matrix.setLabel('Energy resolution\nmatrix')
            self.set_input(eres.matrix.Edges, 'eres_matrix', clone=0)
            self.set_output(eres.matrix.FakeMatrix, 'eres_matrix')

            for i, it in enumerate(self.idx.iterate()):
                trans = eres.add()
                label = it.current_format('Energy resolution:\n{autoindexnd}')
                trans.setLabel(label)

                self.set_input(trans.inputs[0], 'eres', it, clone=0)
                self.set_output(trans.outputs[0], 'eres', it)

    def define_variables(self):
        descriptions=[
                'spatial/temporal resolution',
                'photon statistics',
                'dark noise'
                ]
        for i, (name, unc) in enumerate(self.cfg.pars.items()):
            par = self.common_namespace.reqparameter(name, cfg=unc)
            par.setLabel( 'Energy resolution ({})'.format(descriptions[i]) )

