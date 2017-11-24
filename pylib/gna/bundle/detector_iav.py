# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from gna.bundle import *

@declare_bundle('iav_db_root_v01')
class detector_iav_db_root_v01(TransformationBundle):
    iavmatrix=None
    name = 'iav'
    def __init__(self, **kwargs):
        super(detector_iav_db_root_v01, self).__init__( **kwargs )

        self.parname = kwargs.pop( 'parname', 'OffdiagScale' )

    def build_mat(self):
        """Assembles a chain for IAV detector effect using input matrix"""
        ndiag = self.cfg.get( 'ndiag', 1 )

        norm = self.iavmatrix.sum( axis=0 )
        norm[norm==0.0]=1.0
        self.iavmatrix/=norm

        points = self.storage['matrix'] = C.Points( self.iavmatrix )

        self.output = ()
        for ns in self.namespaces:
            with ns:
                lstorage = self.storage( 'iav_%s'%ns.name )
                renormdiag = R.RenormalizeDiag( ndiag, 1, 1, self.parname )
                lstorage['renormdiag'] = renormdiag
                renormdiag.renorm.inmat( points.points )

                esmear = lstorage['esmear'] = R.HistSmear(True)
                esmear.smear.inputs.SmearMatrix( renormdiag.renorm )
                self.output+=esmear,

        return self.output

    def build(self):
        from file_reader import read_object_auto
        self.iavmatrix = read_object_auto( self.cfg.filename, self.cfg.matrixname, convertto='array' )

        return self.build_mat()

    def define_variables(self):
        if self.cfg.uncertainty_type!='relative':
            raise Exception( 'IAV uncertainty should be relative by definition' )
        for ns in self.namespaces:
            ns.reqparameter( self.parname, central=1.0, relsigma=self.cfg.uncertainty )
