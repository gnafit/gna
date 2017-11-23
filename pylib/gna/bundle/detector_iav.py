# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from gna.bundle import *

@declare_bundle('iav_db_root_v01')
class detector_iav(TransformationBundle):
    def __init__(self, edges, **kwargs):
        kwargs.setdefault( 'storage_name', 'iav')
        super(detector_dbchain, self).__init__( **kwargs )

        self.parname = kwargs.pop( 'parname', 'OffdiagScale' )

    def build_mat(self):
        """Assembles a chain for IAV detector effect using input matrix"""
        ndiag = self.cfg.get( 'ndiag', 1 )

        norm = mat.sum( axis=0 )
        norm[norm==0.0]=1.0
        mat/=norm

        points = self.storage['matrix'] = C.Points( self.iavmatrix )

        output = ()
        for ns in self.namespaces:
            with ns:
                lstorage = self.storage( 'iav_%s'%ns.name )
                renormdiag = R.RenormalizeDiag( ndiag, 1, 1, parname )
                lstorage['renormdiag'] = renormdiag
                renormdiag.renorm.inmat( points.points )

                esmear = lstorage['esmear'] = R.HistSmear(True)
                esmear.smear.inputs.SmearMatrix( renormdiag.renorm )
                output+=(esmear,)

        return output

    def build(self):
        """Assembles a chain for IAV detector effect using input matrix from a file
        see detector_iav() for options"""
        from file_reader import read_object_auto
        self.iavmatrix = read_object_auto( filename, matrixname, convertto='array' )

        return self.build_mat()
