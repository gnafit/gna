# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_eres_common3_v02(TransformationBundleLegacy):
    common_matrix=True
    def __init__(self, *args, **kwargs):
        super(detector_eres_common3_v02, self).__init__( *args, **kwargs )
        self.transformations_in = self.transformations_out

        self.init_indices()

    def build(self):
        with self.common_namespace:
            expose_matrix = self.cfg.get('expose_matrix', False)
            eres = self.eres = R.EnergyResolution(C.stdvector(self.names), expose_matrix, ns=self.common_namespace)

            eres.matrix.setLabel('Energy resolution\nmatrix')
            self.set_input(eres.matrix.Edges, 'eres_matrix', clone=0)
            self.set_output(eres.matrix.FakeMatrix, 'eres_matrix')

            trans = eres.smear
            for i, it in enumerate(self.idx.iterate()):
                if i:
                    trans = eres.add_transformation()
                    eres.add_input()
                label = it.current_format('Energy resolution\n{autoindex}')
                trans.setLabel(label)

                self.set_input(trans.inputs.back(), 'eres', it, clone=0)
                self.set_output(trans.outputs.back(), 'eres', it)

    def define_variables(self):
        descriptions=[
                'spatial/temporal resolution',
                'photon statistics',
                'dark noise'
                ]
        self.names=[]
        for i, (name, unc) in enumerate(self.cfg.pars.items(nested=True)):
            name = '.'.join(name)
            self.names.append(name)
            par = self.common_namespace.reqparameter(name, cfg=unc)
            par.setLabel( 'Energy resolution ({})'.format(descriptions[i]) )

