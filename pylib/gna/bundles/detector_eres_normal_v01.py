# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_eres_normal_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

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
        return (cfg.parameter,), ('eres_matrix', 'eres')

    def build(self):
        expose_matrix = self.cfg.get('expose_matrix', False)
        split_transformations = self.cfg.get('split_transformations', True)

        self.objects = []
        for it_major in self.nidx_major:
            vals = it_major.current_values(name=self.cfg.parameter)
            names = [ '.'.join(vals+(name,)) for name in self.names ]

            eres = C.EnergyResolution(names, expose_matrix, ns=self.namespace)
            self.objects.append(eres)

            self.set_label(eres.matrix, 'matrix', it_major, 'Energy resolution\nmatrix ({autoindex})')
            self.set_input('eres_matrix', it_major, eres.matrix.Edges, argument_number=0)
            self.set_output('eres_matrix', it_major, eres.matrix.FakeMatrix)

            if not split_transformations:
                self.set_label(eres.smear, 'smear', it_major, 'Energy resolution\n({autoindex})')

            trans = eres.smear
            for i, it_minor in enumerate(self.nidx_minor):
                it = it_major + it_minor
                if i:
                    if split_transformations:
                        trans = eres.add_transformation()
                eres.add_input()

                if split_transformations:
                    self.set_label(trans, 'smear', it, 'Energy resolution\n({autoindex})')

                self.set_input('eres', it, trans.inputs.back(), argument_number=0)
                self.set_output('eres', it, trans.outputs.back())

    def define_variables(self):
        descriptions=[
                'spatial/temporal resolution',
                'photon statistics',
                'dark noise'
                ]

        parname = self.cfg.parameter
        parscfg = self.cfg.pars
        labelfmt = self.cfg.get('label', '')
        self.names = None

        for it_major in self.nidx_major:
            major_values = it_major.current_values()
            pars = parscfg[major_values]

            if self.names is None:
                self.names = tuple(sorted(pars.keys()))
            else:
                assert self.names == tuple(sorted(pars.keys()))

            for i, name in enumerate(self.names):
                unc = pars[name]
                it=it_major

                # print(i, name, parname, unc)
                par = self.reqparameter(parname, it, cfg=unc, extra=name)
                label = it.current_format(labelfmt, description=descriptions[i]) if labelfmt else descriptions[i]
                self.set_label(par, 'parameter', it_major, '{description} {autoindex}', name=name, description=descriptions[i])


