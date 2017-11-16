# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from gna.env import namespace, env

def detector_iav( mat, *args, **kwargs  ):
    """Assembles a chain for IAV detector effect using input matrix"""
    gstorage = kwargs.pop( 'storage', None )
    if gstorage:
        storage = gstorage( 'iav' )
    else:
        storage = namespace( None, 'iav' )

    ndiag = kwargs.pop( 'ndiag', 1 )
    parname=  kwargs.pop( 'parname', None )
    namespaces=kwargs.pop( 'namespaces', [env.globalns] )

    norm = mat.sum( axis=0 )
    norm[norm==0.0]=1.0
    mat/=norm

    points = storage['matrix'] = C.Points( mat )

    output = []
    for ns in namespaces:
        with ns:
            lstorage = storage( 'iav_%s'%ns.name )
            if parname:
                renormdiag = R.RenormalizeDiag( ndiag, 1, 1, parname )
            else:
                renormdiag = R.RenormalizeDiag( ndiag, 1, 1 )
            lstorage['renormdiag'] = renormdiag
            renormdiag.renorm.inmat( points.points )

            esmear = lstorage['esmear'] = R.HistSmear( True )
            esmear.smear.inputs.SmearMatrix( renormdiag.renorm )
            output.append( esmear )

    return tuple(output), storage

def detector_iav_from_file( filename, name, *args, **kwargs ):
    """Assembles a chain for IAV detector effect using input matrix from a file
    see detector_iav() for options"""
    from file_reader import read_object_auto
    mat = read_object_auto( filename, name, convertto='array' )

    return detector_iav( mat, *args, **kwargs )
