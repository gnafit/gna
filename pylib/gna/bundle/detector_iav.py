# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C

def detector_iav( mat, *args, **kwargs  ):
    """Assembles a chain for IAV detector effect using input matrix"""
    ndiag = kwargs.pop( 'ndiag', 1 )

    mat/=mat.sum( axis=0 )

    points = C.Points( mat )
    renormdiag = R.RenormalizeDiag( ndiag, int(opts.offdiag), int(opts.upper) )
    renormdiag.renorm.inmat( points.points )

    esmear = R.HistSmear( opts.upper )
    esmear.smear.inputs.SmearMatrix( renormdiag.renorm )

    return esmear, dict( points=points, renormdiag=renormdiag, esmear=esmear )

def detector_iav_from_file( filename, name, *args, **kwargs ):
    """Assembles a chain for IAV detector effect using input matrix from a file
    see detector_iav() for options"""
    from file_reader import read_object_auto
    mat = read_object_auto( filename, name, convertto='array' )

    return detector_iav( mat, *args, **kwargs )
