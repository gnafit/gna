# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *
from collections import OrderedDict

class detector_iav_db_root_v01(TransformationBundle):
    iavmatrix=None
    def __init__(self, **kwargs):
        super(detector_iav_db_root_v01, self).__init__( **kwargs )
        self.transformations_in = self.transformations_out

    def build_mat(self):
        """Assembles a chain for IAV detector effect using input matrix"""
        ndiag = self.cfg.get( 'ndiag', 1 )

        norm = self.iavmatrix.sum( axis=0 )
        norm[norm==0.0]=1.0
        self.iavmatrix/=norm

        points = C.Points( self.iavmatrix, ns=self.common_namespace )
        self.objects['matrix'] = points

        with self.common_namespace:
            for ns in self.namespaces:
                renormdiag = R.RenormalizeDiag( ndiag, 1, 1, self.pars[ns.name], ns=ns )
                renormdiag.renorm.inmat( points.points )

                esmear = R.HistSmear(True)
                esmear.smear.inputs.SmearMatrix( renormdiag.renorm )

                """Save transformations"""
                self.transformations_out[ns.name] = esmear.smear
                self.inputs[ns.name]              = esmear.smear.Ntrue
                self.outputs[ns.name]             = esmear.smear.Nvis

                self.objects[('renormdiag',ns.name)] = renormdiag
                self.objects[('esmear',ns.name)]     = esmear

                """Define observables"""
                ns.addobservable('iav', esmear.smear.Nvis, ignorecheck=True)

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
        for ns in self.namespaces:
            parname = self.cfg.parname.format(ns.name)
            par = self.common_namespace.reqparameter( parname, cfg=self.cfg.scale )
            self.pars[ns.name]=parname

