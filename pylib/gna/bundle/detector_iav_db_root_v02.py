# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *
from collections import OrderedDict

class detector_iav_db_root_v02(TransformationBundle):
    iavmatrix=None
    def __init__(self, **kwargs):
        super(detector_iav_db_root_v02, self).__init__( **kwargs )
        self.transformations_in = self.transformations_out

        self.init_indices()

    def build_mat(self):
        """Assembles a chain for IAV detector effect using input matrix"""
        ndiag = self.cfg.get( 'ndiag', 1 )

        norm = self.iavmatrix.sum( axis=0 )
        norm[norm==0.0]=1.0
        self.iavmatrix/=norm

        points = C.Points( self.iavmatrix, ns=self.common_namespace )
        points.points.setLabel('IAV matrix\nraw')
        self.objects['matrix'] = points
        self.set_output(points.single(), ('iavmatrix', 'raw'))

        with self.common_namespace:
            idxdet, idxother = self.idx.split( 'd' )
            for itdet in idxdet.iterate():
                parname = itdet.current_format('{name}{autoindex}', name=self.cfg.parname)
                renormdiag = R.RenormalizeDiag( ndiag, 1, 1, parname, ns=self.common_namespace )
                renormdiag.renorm.inmat( points.points )
                renormdiag.renorm.setLabel(itdet.current_format('IAV matrix\n{autoindexnd}'))
                self.set_output( renormdiag.single(), 'iavmatrix', itdet )

                for itother in idxother.iterate():
                    it = itdet+itother
                    esmear = R.HistSmear(True)
                    esmear.smear.inputs.SmearMatrix( renormdiag.renorm )
                    esmear.smear.setLabel(it.current_format('IAV effect\n{autoindexnd}'))
                    self.set_input( esmear.smear.Ntrue, 'iav', it, clone=0 )
                    self.set_output( esmear.single(), 'iav', it )

                    self.objects[('esmear',it.current_format())]     = esmear

                self.objects[('renormdiag',itdet.current_format())] = renormdiag

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
        idx = self.idx.get_sub('d')
        for it in idx.iterate():
            parname = it.current_format('{name}{autoindex}', name=self.cfg.parname)
            par = self.common_namespace.reqparameter(parname, cfg=self.cfg.scale)
            par.setLabel('IAV offdiagonal contribution scale')
            # self.pars[ns.name]=parname

