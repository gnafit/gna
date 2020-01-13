# -*- coding: utf-8 -*-

"""Detector energy resolution

implements 1-parameter energy resolution for a detector with multiple zones
"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class detector_multieres_stats_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')
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
        descriptions=[
                'spatial/temporal resolution',
                'photon statistics',
                'dark noise'
                ]
        self.names = list('abc')

        nph = N.loadtxt(self.cfg.nph)
        if nph.size!=self.nidx_major.get_size():
            raise Exception('Number of input Nph values (%i) is not consistent with dimension (%i)'%(nph.size, self.nidx_major.get_size()))

        if 'rescale_nph' in self.cfg:
            factor = self.cfg.rescale_nph/nph.mean()
            print('Subdetector resolution: rescaling Npe to {} pe at 1 MeV, factor {}'.format(self.cfg.rescale_nph, factor))
            nph*=factor

        par_b = nph**-0.5

        if self.cfg.get('verbose', False):
            print('Nph (%i):'%nph.size, nph)
            print('b:', par_b)

        parname = self.cfg.parameter
        labelfmt = self.cfg.get('label', '')

        for it_major, central in zip(self.nidx_major, par_b):
            major_values = it_major.current_values()

            for i, name in enumerate(self.names):
                it=it_major

                if i==1:
                    par = self.reqparameter(parname, it, central=central, fixed=True, extra=name)
                else:
                    par = self.reqparameter(parname, it, central=0.0, fixed=True, extra=name)

                label = it.current_format(labelfmt, description=descriptions[i]) if labelfmt else descriptions[i]
                self.set_label(par, 'parameter', it_major, '{description} {autoindex}', name=name, description=descriptions[i])

