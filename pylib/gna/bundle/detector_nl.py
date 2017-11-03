# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from mpl_tools.root2numpy get_buffers_graph
from scipy.interp import interp1d

def detector_nl( mat, graphs, edges, *args, **kwargs  ):
    """Assembles a chain for IAV detector effect using input matrix"""
    parname = kwargs.pop( 'parnames', [] )

    newx = edges.data()

    # points = C.Points( mat )
    # if parname:
        # renormdiag = R.RenormalizeDiag( ndiag, 1, 1, parname )
    # else:
        # renormdiag = R.RenormalizeDiag( ndiag, 1, 1 )
    # renormdiag.renorm.inmat( points.points )

    # esmear = R.HistSmear( True )
    # esmear.smear.inputs.SmearMatrix( renormdiag.renorm )

    return esmear, dict( points=points, renormdiag=renormdiag, esmear=esmear )

def interpolate( (x, y), edges ):
    fcn = interp1d( x, y, kind='linear' )
    return fcn( edges )

def detector_nl_from_file( filename, names, *args, **kwargs ):
    """Assembles a chain for NL detector effect using input curves from a file
    see detector_nl() for options"""
    tfile = R.TFile( filename, 'READ' )
    if tfile.IsZombie():
        raise IOError( 'Can not read ROOT file: '+filename )

    graphs = [ tfile.Get( name ) for name in names ]
    if not all( graphs ):
        raise IOError( 'Some objects were not read from file: '+filename )

    graphs = [ get_buffers_graph(g) for g in graphs ]

    return detector_nl( mat, graphs, *args, **kwargs )
