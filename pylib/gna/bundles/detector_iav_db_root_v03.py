# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import OrderedDict

class detector_iav_db_root_v03(TransformationBundle):
    iavmatrix=None
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1,1, 'major')

    @staticmethod
    def _provides(self):
        return (), ('iavmatrix_raw', 'iavmatrix', 'iav')

    def build_mat(self):
        """Assembles a chain for IAV detector effect using input matrix"""
        ndiag = self.cfg.get( 'ndiag', 1 )

        norm = self.iavmatrix.sum( axis=0 )
        norm[norm==0.0]=1.0
        self.iavmatrix/=norm

        points = C.Points(self.iavmatrix, ns=self.namespace, labels='IAV matrix\nraw')
        self.context.objects['matrix'] = points
        self.set_output('iavmatrix_raw', None, points.single())

        for itdet in self.nidx_major:
            parname = itdet.current_format(name=self.cfg.parname)
            renormdiag = R.RenormalizeDiag(ndiag, 1, 1, parname, ns=self.namespace, labels=itdet.current_format('IAV matrix\n{autoindex}'))
            renormdiag.renorm.inmat(points.points)
            self.set_output('iavmatrix', itdet, renormdiag.single())

            for itother in idxother.nidx_minor:
                it = itdet+itother
                esmear = R.HistSmear(True, labels=it.current_format('IAV effect\n{autoindex}')) # True for 'upper'
                esmear.smear.inputs.SmearMatrix(renormdiag.renorm)
                self.set_input('iav', it, esmear.smear.Ntrue, argument_number=0)
                self.set_output('iav', it, esmear.single())

                self.context.objects[it.current_values(name='esmear')] = esmear

            self.context.objects[itdet.current_values(names='renormdiag')] = renormdiag

    def build(self):
        from file_reader import read_object_auto
        self.iavmatrix = read_object_auto( self.cfg.filename, self.cfg.matrixname, convertto='array' )

        return self.build_mat()

    def define_variables(self):
        if self.cfg.scale.mode!='relative':
            raise Exception('IAV uncertainty should be relative by definition')
        if self.cfg.scale.central!=1.0:
            raise exception('IAV scale should be 1 by definition')

        self.pars = OrderedDict()
        for it in self.nidx_major:
            self.reqparameter(self.cfg.parname, it, cfg=self.cfg.scale, label='IAV offdiagonal contribution scale at {autoindex}')

