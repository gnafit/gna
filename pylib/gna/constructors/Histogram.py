#!/usr/bin/env python

from load import ROOT as R
import numpy as np

"""Construct Histogram object from two arrays: edges and data"""
def Histogram(edges, data=None, *args, **kwargs):
    edges = np.ascontiguousarray(edges, dtype='d')
    reqsize = (edges.size-1)
    if data is None:
        data  = np.zeros(reqsize, dtype='d')
    else:
        if reqsize!=data.size:
            raise Exception( 'Bin edges and data are not consistent (%i and %i)'%( edges.size, data.size ) )
        data  = np.ascontiguousarray(data,  dtype='d')

    return R.Histogram( data.size, edges, data, *args, **kwargs )

"""Construct Histogram2d object from two arrays: edges and data"""
def Histogram2d(xedges, yedges, data=None, *args, **kwargs):
    xedges = np.ascontiguousarray(xedges, dtype='d')
    yedges = np.ascontiguousarray(yedges, dtype='d')
    reqsize = (xedges.size-1)*(yedges.size-1)
    if data is None:
        data = np.zeros(reqsize, dtype='d')
    else:
        if reqsize!=data.size:
            raise Exception( 'Bin edges and data are not consistent (%i,%i and %i)'%( xedges.size, yedges.size, data.size ) )
        data = np.ascontiguousarray(data,   dtype='d').ravel(order='F')

    return R.Histogram2d( xedges.size-1, xedges, yedges.size-1, yedges, data, *args, **kwargs )
