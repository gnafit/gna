#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import ROOT as R
import numpy as N
import root2numpy as R2N
from mpl_tools import helpers
from matplotlib import pyplot as P

def plot_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.plot

    executes pyplot.plot(x, y, *args, **kwargs) with first two arguments overridden
    all other arguments passes as is.

    if label=='@auto' replaces plot label with histogram title

    returns pyplot.plot() result
    """
    if kwargs.get( 'label', None )=='@auto':
        kwargs['label'] = h.GetTitle()

    lims=R2N.get_bin_edges_axis( h.GetXaxis() )
    height=R2N.get_buffer_hist1( h ).copy()

    return helpers.plot_hist( lims, height, *args, **kwargs )

def bar_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.bar

    executes pyplot.bar(left, height, width, *args, **kwargs) with first two arguments overridden
    all other arguments passes as is.

    if label=='@auto' replaces plot label with histogram title

    Options:
        divide=N - divide bin width by N
        shift=N  - shift bin left edge by N*width

    returns pyplot.bar() result
    """
    divide = kwargs.pop( 'divide', None )
    shift  = kwargs.pop( 'shift', 0 )
    if kwargs.get( 'label', None )=='@auto':
        kwargs['label'] = h.GetTitle()

    height=R2N.get_buffer_hist1( h ).copy()
    ax=h.GetXaxis()
    lims, fixed=R2N.get_bin_edges_axis( ax, type=True )
    width=None
    left  = lims[:-1]
    if fixed:
        width = ax.GetBinWidth( 1 )
    else:
        width = lims[1:] - left

    if divide:
        width/=divide
        left+=width*shift

    return P.bar( left, height, width, *args, **kwargs )

def errorbar_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.errorbar

    executes pyplot.errorbar(x, y, yerr, xerr, *args, **kwargs) with first four arguments overridden
    all other arguments passes as is.

    if label=='@auto' replaces plot label with histogram title

    Uses histgram's errors if they are defined. Uses sqrt(N) otherwise.

    returns pyplot.errorbar() result
    """
    if kwargs.get( 'label', None )=='@auto':
        kwargs['label'] = h.GetTitle()

    noyerr, mask, = [ kwargs.pop(x) if x in kwargs else None for x in ['noyerr', 'mask'] ]
    centers = R2N.get_bin_centers_axis( h.GetXaxis())
    hwidths = R2N.get_bin_widths_axis( h.GetXaxis())*0.5
    height=R2N.get_buffer_hist1( h ).copy()
    if not mask is None:
        height = N.ma.array( height, mask=mask )

    yerr = None
    if not noyerr:
        yerr2 = R2N.get_err_buffer_hist1( h )
        if yerr2 is None:
            yerr2 = height
        yerr = yerr2**0.5

    if not 'fmt' in kwargs:
        kwargs['fmt']='none'
    return P.errorbar( centers, height, yerr, hwidths, *args, **kwargs )

def pcolormesh_hist2( h, *args, **kwargs ):
    """Plot 2-dimensinal histogram using pyplot.pcolormesh

    executes pyplot.pcolormesh(x, y, C, *args, **kwargs) with first two arguments overridden
    all other arguments passes as is.

    Options:
        mask=F - exclude value F from plotting (set mask=0 to avoid plotting 0.0)
        colorbar - if true, plot colorbar with height aligned to the axes height

    returns pyplot.pcolormesh()[, pyplot.colorbar] result
    """
    mask = kwargs.pop( 'mask', None )
    colorbar = kwargs.pop( 'colorbar', None )

    x1 = R2N.get_bin_edges_axis(h.GetXaxis())
    y1 = R2N.get_bin_edges_axis(h.GetYaxis())

    x, y = N.meshgrid( x1, y1 )

    buf = R2N.get_buffer_hist2( h, mask=mask ).copy()

    res = P.pcolormesh( x, y, buf, *args, **kwargs )
    if colorbar:
        cbar = helpers.add_colorbar( res )
        return res, cbar

    return res

def pcolorfast_hist2( h, *args, **kwargs ):
    """Plot 2-dimensinal histogram using ax.pcolorfast

    executes ax.pcolorfast(x, y, C, *args, **kwargs) with first two arguments overridden
    all other arguments passes as is.

    Options:
        mask=F - exclude value F from plotting (set mask=0 to avoid plotting 0.0)
        colorbar - if true, plot colorbar with height aligned to the axes height

    returns ax.pcolorfast()[, pyplot.colorbar] result
    """
    mask = kwargs.pop( 'mask', None )
    colorbar = kwargs.pop( 'colorbar', None )

    xax = h.GetXaxis()
    yax = h.GetYaxis()
    if xax.GetXbins().GetSize()>0 or yax.GetXbins().GetSize()>0:
        print( 'Can not draw 2D a histogram with variable bin widths' )
        print( 'Use pcolormesh method instead' )
        return
    x = [ xax.GetXmin(), xax.GetXmax() ]
    y = [ yax.GetXmin(), yax.GetXmax() ]

    buf = R2N.get_buffer_hist2( h,  mask=mask ).copy()

    ax = P.gca()
    res = ax.pcolorfast( x, y, buf, *args, **kwargs )

    if colorbar:
        cbar = helpers.add_colorbar( res )
        return res, cbar

    return res

def imshow_hist2( h, *args, **kwargs ):
    """Plot 2-dimensinal histogram using pyplot.imshow

    executes pyplot.imshow(x, y, C, *args, **kwargs) with first two arguments overridden
    all other arguments passes as is.

    Options:
        mask=F - exclude value F from plotting (set mask=0 to avoid plotting 0.0)
        colorbar - if true, plot colorbar with height aligned to the axes height

    returns pyplot.imshow()[, pyplot.colorbar] result
    """
    mask = kwargs.pop( 'mask', None )
    colorbar = kwargs.pop( 'colorbar', None )

    xax = h.GetXaxis()
    yax = h.GetYaxis()
    if xax.GetXbins().GetSize()>0 or yax.GetXbins().GetSize()>0:
        print( 'Can not draw 2D a histogram with variable bin widths' )
        print( 'Use pcolormesh method or draweHist2Dmesh function instead' )
        return
    extent = [ xax.GetXmin(), xax.GetXmax(), yax.GetXmin(), yax.GetXmax()  ]

    buf = R2N.get_buffer_hist2( h,  mask=mask ).copy()

    res = P.imshow( buf, *args, extent=extent, **kwargs )
    if colorbar:
        cbar = helpers.add_colorbar( res )
        return res,cbar

    return res

def bind():
    setattr( R.TH1, 'bar', bar_hist1 )
    setattr( R.TH1, 'errorbar', errorbar_hist1 )
    setattr( R.TH1, 'plot', plot_hist1 )

    setattr( R.TH2, 'pcolorfast', pcolorfast_hist2 )
    setattr( R.TH2, 'pcolormesh', pcolormesh_hist2 )
    setattr( R.TH2, 'imshow', imshow_hist2 )


