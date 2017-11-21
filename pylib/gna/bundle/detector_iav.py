# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from gna.env import namespace, env

def detector_iav( mat, *args, **kwargs  ):
    """Assembles a chain for IAV detector effect using input matrix"""
    cfg = kwargs.pop( 'cfg', kwargs )
    ndiag   = cfg.get( 'ndiag', 1 )

    parname = kwargs.pop( 'parname', 'OffdiagScale' )
    gstorage = kwargs.pop( 'storage', None )
    if gstorage:
        storage = gstorage( 'iav' )
    else:
        storage = namespace( None, 'iav' )

    namespaces=kwargs.pop( 'namespaces', [env.globalns] )

    norm = mat.sum( axis=0 )
    norm[norm==0.0]=1.0
    mat/=norm

    points = storage['matrix'] = C.Points( mat )

    output = []
    for ns in namespaces:
        with ns:
            lstorage = storage( 'iav_%s'%ns.name )
            renormdiag = R.RenormalizeDiag( ndiag, 1, 1, parname )
            lstorage['renormdiag'] = renormdiag
            renormdiag.renorm.inmat( points.points )

            esmear = lstorage['esmear'] = R.HistSmear( True )
            esmear.smear.inputs.SmearMatrix( renormdiag.renorm )
            output.append( esmear )

    return tuple(output), storage

def detector_iav_from_file( filename, matrixname, *args, **kwargs ):
    """Assembles a chain for IAV detector effect using input matrix from a file
    see detector_iav() for options"""
    from file_reader import read_object_auto
    mat = read_object_auto( filename, matrixname, convertto='array' )

    return detector_iav( mat, *args, **kwargs )
