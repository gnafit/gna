# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from converters import convert
from mpl_tools.root2numpy import get_buffers_graph
from scipy.interpolate import interp1d
from matplotlib import pyplot as P
from gna.env import env, namespace

def detector_nl( graphs, edges, *args, **kwargs  ):
    """Assembles a chain for IAV detector effect using input matrix"""
    cfg = kwargs.pop( 'cfg', kwargs )
    names = cfg['names']

    gstorage = kwargs.pop( 'storage', None )
    if gstorage:
        storage = gstorage( 'nonlinearity' )
    else:
        storage = namespace( None, 'nonlinearity' )

    debug = kwargs.pop( 'debug', False )
    namespaces = kwargs.pop( 'namespaces', [env.globalns] )

    #
    # Interpolate curves on the default binning
    # (extrapolate as well)
    #
    storage['edges'] = edges
    newx = edges.data()
    storage('inputs')['edges'] = newx
    newy = []
    for xy, name in zip(graphs, names):
        f = interpolate( xy, newx )
        newy.append(f)
        storage('inputs')[name] = f.copy()

    #
    # All curves but first are the corrections to the nominal
    #
    for f in newy[1:]:
        f-=newy[0]

    #
    # Correlated part of the energy nonlinearity factor
    # a weighted sum of input curves
    #
    corr_lsnl = storage['lsnl_factor'] = R.WeightedSum( convert(names, 'stdvector') )
    for y, name in zip( newy, names ):
        pts = C.Points( y )
        storage('curves')[name] = pts
        corr_lsnl.sum[name]( pts )

    output = []
    labels = convert(['escale'], 'stdvector')
    for i, ns in enumerate(namespaces):
        with ns:
            #
            # Uncorrelated between detectors part of the energy nonlinearity factor
            # correlated part multiplicated by the scale factor
            #
            lstorage = storage('escale_%s'%ns.name if ns.name else 'escale')
            corr = lstorage['factor'] = R.WeightedSum( labels, labels )
            corr.sum['escale']( corr_lsnl.sum )

            #
            # Finally, original bin edges multiplied by the correction factor
            #
            newe = lstorage['edges_mod'] = R.Product()
            newe.multiply( edges )
            newe.multiply( corr.sum )

            #
            # Construct the nonlinearity calss
            #
            nonlin = lstorage['nonlinearity'] = R.HistNonlinearity( debug )
            nonlin.set( edges, newe.product )
            output.append( nonlin )

    return tuple(output), storage

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
