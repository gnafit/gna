# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import OrderedDict

class detector_energy_leakage_root_v01(TransformationBundle):
    """Energy distortion due to the energy leakage v01, defined via a histogram from a ROOT file

    changes since detector_iav_db_root_v03:
        - uncertainty is optional
        - major index may be absent

    Configuration fields:
        - filename
        - matrixname

    Optional configuration fields:
        - parname - nuisance parameter name (string)
        - scale - 'uncertain' with the definition of uncertainty
        - ndiag=1 - number of diagonals to apply scale factor (with uncertainty) to

    Provided variables:
        - parname - uncertain parameter name

    Provided outputs:
        - 'eleak_matrix_raw' - raw energy leakage matrix (in case of nuisance parameter)
        - 'eleak_matrix' - energy leakage matrix after scaling the diagonal
        - 'eleak' - the result of the distortion

    Provided inputs:
        - 'eleak' - an input to apply the distortion
    """
    eleak_matrix=None
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, 'major')

        paroptnames = 'parname', 'scale', 'ndiag'
        paropts = (s in self.cfg for s in paroptnames)
        if any(paropts) and not all(paropts):
            self.exception('Not all options are specified. Need: {!s}'.format(paroptnames))

    @staticmethod
    def _provides(cfg):
        parname = cfg.get('parname')
        if parname:
            return (parname,), ('eleak_matrix_raw', 'eleak_matrix', 'eleak')

        return (), ('eleak_matrix', 'eleak')

    def build_mat(self):
        """Assembles a chain for eleak detector effect using input matrix"""
        haspar = bool(self.cfg.get('parname'))

        ndiag = self.cfg.get('ndiag', 1)

        norm = self.eleak_matrix.sum( axis=0 )
        norm[norm==0.0]=1.0
        self.eleak_matrix/=norm

        points = C.Points(self.eleak_matrix, ns=self.namespace, labels='Eleak matrix\n raw')
        self.context.objects['matrix'] = points
        if haspar:
            self.set_output('eleak_matrix_raw', None, points.single())
        else:
            self.set_output('eleak_matrix', None, points.single())

        for itdet in self.nidx_major:
            if haspar:
                parname = itdet.current_format(name=self.cfg.parname)
                # Target=OffDiagonal, Mode=Upper
                renormdiag = R.RenormalizeDiag(ndiag, 1, 1, parname, ns=self.namespace, labels=itdet.current_format('Eleak matrix\n {autoindex}'))
                points.points >> renormdiag.renorm.inmat
                self.set_output('eleak_matrix', itdet, renormdiag.single())
                self.context.objects[itdet.current_values(name='renormdiag')] = renormdiag
                matinput = renormdiag.renorm
            else:
                matinput = points.points

            for itother in self.nidx_minor:
                it = itdet+itother
                esmear = R.HistSmear(True, labels=it.current_format('{{Eleak effect|{autoindex}}}')) # True for 'upper'
                matinput >> esmear.smear.inputs.SmearMatrix
                self.set_input('eleak', it, esmear.smear.Ntrue, argument_number=0)
                self.set_output('eleak', it, esmear.single())

                self.context.objects[it.current_values(name='esmear')] = esmear

    def build(self):
        from tools.data_load import read_object_auto
        res = read_object_auto(self.cfg.filename, name=self.cfg.matrixname, convertto='array')
        if isinstance(res, tuple):
            self.eleak_matrix = res[-1]
        else:
            self.eleak_matrix = res

        return self.build_mat()

    def define_variables(self):
        parname = self.cfg.get('parname')
        if not parname:
            return

        scale = self.cfg.scale
        if scale.mode!='relative':
            raise Exception('Eleak uncertainty should be relative by definition')
        if scale.central!=1.0:
            raise exception('Eleak scale should be 1 by definition')

        for it in self.nidx_major:
            self.reqparameter(parname, it, cfg=scale, label='Eleak offdiagonal contribution scale at {autoindex}')

