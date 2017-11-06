# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from converters import convert
from mpl_tools.root2numpy import get_buffers_graph
from scipy.interpolate import interp1d
from matplotlib import pyplot as P

def detector_nl( graphs, edges, *args, **kwargs  ):
    """Assembles a chain for IAV detector effect using input matrix"""
    names = kwargs.pop( 'names' )
    debug = kwargs.pop( 'debug', False )
    transf = dict( curves={}, inputs={} )
    nonlin = transf['nonlinearity'] = R.HistNonlinearity( debug )

    #
    # Interpolate on the default binning
    # (extrapolate as well)
    #
    transf['edges'] = edges
    newx = edges.data()
    transf['inputs']['edges'] = newx
    newy = []
    for xy, name in zip(graphs, names):
        f = interpolate( xy, newx )
        newy.append(f)
        transf['inputs'][name] = f.copy()

    #
    # All curves but first are the corrections to the nominal
    #
    for f in newy[1:]:
        f-=newy[0]

    wsum = transf['sum'] = R.WeightedSum( convert(names, 'stdvector') )
    for y, name in zip( newy, names ):
        pts = C.Points( y )
        transf['curves'][name] = pts
        wsum.sum[name]( pts )

    newe = transf['newe'] = R.Product()
    newe.multiply( edges )
    newe.multiply( wsum.sum )
    nonlin.set( edges, newe.product )

    return nonlin, transf

def interpolate( (x, y), edges):
    fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value='extrapolate' )
    res = fcn( edges )

    return res

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

    return detector_nl( graphs, *args, names=names, **kwargs )
