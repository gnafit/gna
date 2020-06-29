#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import ROOT as R
import numpy as N

def get_bin_edges_hist1(h):
    edges = N.array([h.GetBinLowEdge(i)+h.GetBinWidth(i) for i in range(h.GetNbinsX()+1)])
    return edges

def get_buffer_hist1( h, flows=False ):
    """Return TH1* histogram data buffer
    if flows=False, exclude underflow and overflow
    """
    buf = h.GetArray()
    buf = N.frombuffer(buf , N.dtype( buf.typecode ), h.GetNbinsX()+2 )
    if not flows: buf = buf[1:-1]
    return buf

def get_buffers_hist1(h):
    """Get X/Y buffers of 1D histogram"""
    return get_bin_centers_axis(h.GetXaxis()), get_buffer_hist1(h, flows=False)

def get_buffers_graph_or_hist1(obj):
    """Get X/Y buffers of eighter 1D histogram or TGraph"""
    if isinstance(obj, R.TGraph):
        return get_buffers_graph(obj)
    if isinstance(obj, R.TH1) and obj.GetDimension()==1:
        return get_buffers_hist1(obj)

    raise TypeError('The object is not TH1/TGraph: {!s}'.format(obj))

def get_err_buffer_hist1( h, flows=False ):
    """Return TH1* histogram error buffer
    if flows=False, exclude underflow and overflow
    """
    sw2 = h.GetSumw2()
    if sw2.GetSize()==0: return None

    buf = N.frombuffer( sw2.GetArray(), 'd', h.GetNbinsX()+2 )
    if not flows: buf = buf[1:-1]
    return buf

def get_buffer_hist2( h, **kwargs ):
    """Return histogram data buffer
    if flows=False, exclude underflow and overflow
    if mask==0.0 than bins with 0.0 content will be masked
    NOTE: buf[biny][binx] is the right access signature
    """
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    buf = h.GetArray()
    res = N.frombuffer( buf, N.dtype( buf.typecode ), (nx+2)*(ny+2) ).reshape( ( ny+2, nx+2 ) )
    if not kwargs.pop( 'flows', False ):
        res = res[1:ny+1,1:nx+1]

    if kwargs.pop( 'asmatrix', False ):
        res = numpy.matrix( res )

    mask = kwargs.pop( 'mask', None )
    if not mask is None:
        res = N.ma.array( res, mask = res==mask )

    return res

def get_err_buffer_hist2( h, **kwargs ):
    """Return histogram error buffer
    if flows=False, exclude underflow and overflow
    if mask==0.0 than bins with 0.0 content will be masked
    NOTE: buf[biny][binx] is the right access signature
    """
    sw2 = h.GetSumw2()
    if sw2.GetSize()==0: return None
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    buf = sw2.GetArray()
    res = N.frombuffer( buf, N.dtype( buf.typecode ), (nx+2)*(ny+2) ).reshape( ( ny+2, nx+2 ) )

    if not kwargs.pop( 'flows', False ):
        res = res[1:ny+1,1:nx+1]

    return res

def get_bin_edges_axis( ax, type=False, rep=None ):
    """Get the array with bin edges for TAxis
    returns also whether the bins are fixed if type=True
    """
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    lims = None
    fixed = False
    if n>0:
        lims = N.frombuffer( xbins.GetArray(), N.double, n )
        fixed = False
    else:
        lims = N.linspace( ax.GetXmin(), ax.GetXmax(), ax.GetNbins()+1 )
        fixed = True
    if rep and rep>1:
        res = [ lims ]
        delta = -lims[0]
        for i in xrange( rep-1 ):
            res.append( res[-1][-1] + lims[1:] + delta )
        lims = N.concatenate( res )

    if type: return lims, fixed
    return lims

def get_bin_centers_axis( ax ):
    """Get the array with bin centers"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n>0:
        lims = N.frombuffer( xbins.GetArray(), N.double, n )
        return ( lims[:-1] + lims[1:] )*0.5
    hwidth = ax.GetBinWidth(1)*0.5
    return N.linspace( ax.GetXmin()+hwidth, ax.GetXmax()-hwidth, ax.GetNbins() )

def get_bin_widths_axis( ax ):
    """Get the array with bin widths or bin width if it's constant"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n>0:
        lims = N.frombuffer( xbins.GetArray(), N.double, n )
        return ( lims[1:] - lims[:-1] )
    return ax.GetBinWidth(1)

def get_buffers_graph( g ):
    """Get TGraph x and y buffers"""
    npoints = g.GetN()
    if npoints:
        # return N.frombuffer(g.GetX(), dtype=N.double, count=npoints), \
               # N.frombuffer(g.GetY(), dtype=N.double, count=npoints)
        return N.array(g.GetX(), dtype=N.double), \
               N.array(g.GetY(), dtype=N.double)

    return None, None

def get_err_buffers_graph( g ):
    """Get TGraphErrors x and y error buffers"""
    npoints = g.GetN()
    if npoints:
        # return N.frombuffer(g.GetEX(), dtype=N.double, count=npoints), \
               # N.frombuffer(g.GetEY(), dtype=N.double, count=npoints)
        return N.array(g.GetEX(), dtype=N.double), \
               N.array(g.GetEY(), dtype=N.double)

    return None, None

def get_err_buffers_graph_asymm( g ):
    """Get TGraphAsymError x and y error buffers"""
    npoints = g.GetN()
    if npoints:
        return N.frombuffer(g.GetEXlow(),  dtype=N.double, count=npoints), \
               N.frombuffer(g.GetEXhigh(), dtype=N.double, count=npoints), \
               N.frombuffer(g.GetEYlow(),  dtype=N.double, count=npoints), \
               N.frombuffer(g.GetEYhigh(), dtype=N.double, count=npoints)
        # return N.array(g.GetEXlow(),  dtype=N.double), \
               # N.array(g.GetEXhigh(), dtype=N.double), \
               # N.array(g.GetEYlow(),  dtype=N.double), \
               # N.array(g.GetEYhigh(), dtype=N.double)

    return None, None, None, None

def get_buffer_matrix( m, **kwargs ):
    """Get TMatrix buffer
    if mask==0.0 than cells with 0.0 content will be masked
    """
    mask = kwargs.pop( 'mask', None )
    cbuf = m.GetMatrixArray()
    res = N.frombuffer( cbuf, N.dtype( cbuf.typecode ), m.GetNoElements() ).reshape( m.GetNrows(), m.GetNcols() )
    if not mask is None:
        res = N.ma.array( res, mask = res==mask )
    return res

def bind():
    """Bind functions to ROOT classes"""
    setattr( R.TH1, 'get_buffer',     get_buffer_hist1 )
    setattr( R.TH1, 'get_err_buffer', get_err_buffer_hist1 )
    setattr( R.TH1, 'get_edges', get_bin_edges_hist1 )

    setattr( R.TH2, 'get_buffer',     get_buffer_hist2 )
    setattr( R.TH2, 'get_err_buffer', get_err_buffer_hist2 )

    setattr( R.TAxis, 'get_bin_edges',  get_bin_edges_axis )
    setattr( R.TAxis, 'get_bin_widths', get_bin_widths_axis )

    setattr( R.TGraph,            'get_buffers',     get_buffers_graph )
    setattr( R.TGraphErrors,      'get_err_buffers', get_err_buffers_graph )
    setattr( R.TGraphAsymmErrors, 'get_err_buffers', get_err_buffers_graph_asymm )

    setattr( R.TMatrixD, 'get_buffer', get_buffer_matrix )
    setattr( R.TMatrixF, 'get_buffer', get_buffer_matrix )

data_handlers = {R.TH1: get_buffer_hist1,
                 R.TH2: get_buffer_hist2,
                 R.TAxis: get_bin_edges_axis,
                 R.TGraph: get_buffers_graph,
                 }
err_handlers = {}
