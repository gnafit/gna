#!/usr/bin/env python

from load import ROOT as R
import numpy as np

"""Construct Histogram object from two arrays (edges and data) or from TH1"""
def Histogram(arg1, arg2=None, /, *args, **kwargs):
    if arg2 is None:
        if isinstance(arg1, R.TH1):
            edges = arg1.GetXaxis().get_bin_edges().astype('d')
            data  = np.ascontiguousarray(arg1.get_buffer().T, dtype='d') # Histogram buffer is transposed (y, x)
        else:
            edges = np.ascontiguousarray(arg1, dtype='d')
            data  = np.zeros(edges.size-1, dtype='d')
    else:
        edges = np.ascontiguousarray(arg1, dtype='d')
        data  = np.ascontiguousarray(arg2, dtype='d')

        if (edges.size-1)!=data.size:
            raise Exception('Bin edges and data are not consistent (%i and %i)'%(edges.size, data.size))

    return R.Histogram(data.size, edges, data, *args, **kwargs)

"""Construct Histogram2d object from two arrays: edges and data"""
def Histogram2d(arg1, arg2=None, arg3=None, /, *args, **kwargs):
    if arg2 is None and arg3 is None:
        if isinstance(arg1, R.TH2):
            xedges = arg1.GetXaxis().get_bin_edges().astype('d')
            yedges = arg1.GetYaxis().get_bin_edges().astype('d')
            data   = np.ascontiguousarray(arg1.get_buffer().T, dtype='d').ravel(order='F') # histogram buffer is transposed (y, x)
        else:
            raise Exception('Should provide (xedges, yedges[, data]) or (TH2) to construct Histogram2d')
    else:
        xedges = np.ascontiguousarray(arg1, dtype='d')
        yedges = np.ascontiguousarray(arg2, dtype='d')

        reqsize = (xedges.size-1)*(yedges.size-1)
        if arg3 is None:
            data = np.zeros(reqsize, dtype='d')
        else:
            data = arg3
            if reqsize!=data.size:
                raise Exception( 'Bin edges and data are not consistent (%i,%i and %i)'%( xedges.size, yedges.size, data.size ) )
            data = np.ascontiguousarray(data, dtype='d').ravel(order='F')

    return R.Histogram2d(xedges.size-1, xedges, yedges.size-1, yedges, data, *args, **kwargs)
