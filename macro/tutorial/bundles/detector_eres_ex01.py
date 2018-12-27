# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_eres_ex01(TransformationBundle):
    common_matrix=True
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

    def build(self):
        pass
        # with self.common_namespace:
            # expose_matrix = self.cfg.get('expose_matrix', False)
            # eres = self.eres = R.EnergyResolution(C.stdvector(self.names), expose_matrix, ns=self.common_namespace)

            # eres.matrix.setLabel('Energy resolution\nmatrix')
            # self.set_input(eres.matrix.Edges, 'eres_matrix', clone=0)
            # self.set_output(eres.matrix.FakeMatrix, 'eres_matrix')

            # trans = eres.smear
            # for i, it in enumerate(self.idx.iterate()):
                # if i:
                    # trans = eres.add_transformation()
                    # eres.add_input()
                # label = it.current_format('Energy resolution\n{autoindex}')
                # trans.setLabel(label)

                # self.set_input(trans.inputs.back(), 'eres', it, clone=0)
                # self.set_output(trans.outputs.back(), 'eres', it)

    def define_variables(self):
        descriptions=[
                'spatial/temporal resolution',
                'photon statistics',
                'dark noise'
                ]

        parname = self.cfg.parameter
        parscfg = self.cfg.pars
        labelfmt = self.cfg.get('label', '')

        for it_major in self.nidx_major:
            major_values = it_major.current_values()
            pars = parscfg[major_values]

            for i, (name, unc) in enumerate(pars.items()):
                for it_minor in self.nidx_minor:
                    it=it_major+it_minor

                    label = it.current_format(labelfmt, description=descriptions[i]) if labelfmt else descriptions[i]
                    par = self.reqparameter(parname, it, cfg=unc, label=label, extra=name)


