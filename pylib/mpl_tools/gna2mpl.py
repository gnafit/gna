#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load import ROOT as R
import numpy as N
import root2numpy as R2N
from mpl_tools import helpers
from matplotlib import pyplot as P
from gna.bindings import DataType

def ifNd(output, ndim):
    dtype = output.datatype()
    if dtype.shape.size()==ndim:
        return

    raise Exception('Output is supposed to be %id: '%ndim+str(dtype))

def ifHist(output):
    dtype = output.datatype()
    if dtype.kind==2:
        return

    raise Exception('Output is supposed to be hist: '+str(dtype))

def ifPoints(output):
    dtype = output.datatype()
    if dtype.kind==1:
        return

    raise Exception('Output is supposed to be points: '+str(dtype))

def plot_points( output, *args, **kwargs ):
    """Plot array using pyplot.plot

    executes pyplot.plot(y, *args, **kwargs) with first argument overridden,
    all other arguments are passed as is.

    returns pyplot.plot() result
    """
    if kwargs.pop('transpose', False):
        points=output.data().T.copy()
    else:
        points=output.data().copy()

    return P.plot(points, *args, **kwargs)

def plot_vs_points(outputy, outputx, *args, **kwargs):
    """Plot array using pyplot.plot

    executes pyplot.plot(y, *args, **kwargs) with first argument overridden,
    all other arguments are passed as is.

    returns pyplot.plot() result
    """
    if isinstance(outputx, (N.ndarray, list)):
        pointsx=outputx
    else:
        pointsx=outputx.data().copy()

    if isinstance(outputy, (N.ndarray, list)):
        pointsy=outputy
    else:
        pointsy=outputy.data().copy()

    if kwargs.pop('transpose', False):
        pointsx, pointsy=pointsx.T, pointsy.T

    return P.plot(pointsx, pointsy, *args, **kwargs)

def vs_plot_points(outputx, outputy, *arsg, **kwargs):
    return plot_vs_points(outputy, outputx, *args, **kwargs)

def plot_hist1( output, *args, **kwargs ):
    """Plot 1-dimensinal output using pyplot.plot

    executes pyplot.plot(x, y, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    returns pyplot.plot() result
    """
    ifNd(output, 1)
    ifHist(output)

    height=output.data().copy()
    lims=N.array(output.datatype().edges)

    return helpers.plot_hist( lims, height, *args, **kwargs )

def bar_hist1( output, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.bar

    executes pyplot.bar(left, height, width, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    Options:
        divide=N - divide bin width by N
        shift=N  - shift bin left edge by N*width

    returns pyplot.bar() result
    """
    ifNd(output, 1)
    ifHist(output)
    divide = kwargs.pop( 'divide', None )
    shift  = kwargs.pop( 'shift', 0 )

    height=output.data().copy()
    lims  = N.array(output.datatype().edges)
    left  = lims[:-1]
    width = lims[1:] - left

    if divide:
        width/=divide
        left+=width*shift

    kwargs.setdefault( 'align', 'edge' )
    return P.bar( left, height, width, *args, **kwargs )

# def errorbar_hist1( h, *args, **kwargs ):
    # """Plot 1-dimensinal histogram using pyplot.errorbar

    # executes pyplot.errorbar(x, y, yerr, xerr, *args, **kwargs) with first four arguments overridden
    # all other arguments are passed as is.

    # Uses histgram's errors if they are defined. Uses sqrt(N) otherwise.

    # returns pyplot.errorbar() result
    # """
    #
    # noyerr, mask, = [ kwargs.pop(x) if x in kwargs else None for x in ['noyerr', 'mask'] ]
    # centers = R2N.get_bin_centers_axis( h.GetXaxis())
    # hwidths = R2N.get_bin_widths_axis( h.GetXaxis())*0.5
    # height=R2N.get_buffer_hist1( h ).copy()
    # if not mask is None:
        # height = N.ma.array( height, mask=mask )

    # yerr = None
    # if not noyerr:
        # yerr2 = R2N.get_err_buffer_hist1( h )
        # if yerr2 is None:
            # yerr2 = height
        # yerr = yerr2**0.5

    # if not 'fmt' in kwargs:
        # kwargs['fmt']='none'
    # return P.errorbar( centers, height, yerr, hwidths, *args, **kwargs )


def get_2d_buffer(output, transpose=False, mask=None):
    buf = output.data().copy()

    if mask is not None:
        buf = N.ma.array(buf, mask=buf==mask)

    if transpose:
        buf = buf.T

    return buf

def get_hist2d_data(output, kwargs):
    ifNd(output, 2)
    ifHist(output)

    mask      = kwargs.pop( 'mask', None )
    transpose = kwargs.pop( 'transpose', False )

    dtype=output.datatype()
    xedges, yedges = N.array(dtype.edgesNd[0]), N.array(dtype.edgesNd[1])
    if transpose:
        xedges, yedges=yedges, xedges

    buf = get_2d_buffer(output, transpose=transpose, mask=mask)

    return buf, xedges, yedges

def get_bin_width(edges):
    widths = edges[1:]-edges[:-1]

    status = N.allclose(widths, widths[0])
    if not status:
        print('Widths', widths)
        raise Exception('Bin widths are not equal')

    return widths[0]

def get_hist2d_data_eq(output, kwargs):
    buf, xedges, yedges = get_hist2d_data(output, kwargs)

    xw = get_bin_width(xedges)
    yw = get_bin_width(yedges)

    return buf, xw, xedges, yw, yedges

def colorbar_or_not(res, cbaropt):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, dict):
        cbaropt = {}

    cbar = helpers.add_colorbar(res, **cbaropt)

    return res, cbar

def colorbar_or_not_3d(res, cbaropt, mappable=None, cmap=None):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, dict):
        cbaropt = {}

    cbaropt.setdefault('aspect', 4)
    cbaropt.setdefault('shrink', 0.5)

    if mappable is None:
        cbar = P.colorbar(res, **cbaropt)
    else:
        colourMap = P.cm.ScalarMappable()
        colourMap.set_array(mappable)
        cbar = P.colorbar(colourMap, **cbaropt)

    return res, cbar

def pcolorfast_hist2(output, *args, **kwargs):
    kwargs['transpose'] = ~kwargs.get('transpose', False)
    buf, xe, xedges, yw, yedges = get_hist2d_data_eq(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )
    x = [yedges[0], yedges[-1]]
    y = [xedges[0], xedges[-1]]

    ax = P.gca()
    res = ax.pcolorfast( x, y, buf, *args, **kwargs )

    return colorbar_or_not(res, colorbar)

def pcolormesh_hist2(output, *args, **kwargs):
    buf, xedges, yedges = get_hist2d_data(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )

    x, y = N.meshgrid(xedges, yedges, indexing='ij')

    res = P.pcolormesh(x, y, buf, *args, **kwargs)

    return colorbar_or_not(res, colorbar)

def pcolor_hist2(output, *args, **kwargs):
    buf, xedges, yedges = get_hist2d_data(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )

    x, y = N.meshgrid(xedges, yedges, indexing='ij')

    res = P.pcolor(x, y, buf, *args, **kwargs)

    return colorbar_or_not(res, colorbar)

def imshow_hist2(output, *args, **kwargs):
    kwargs['transpose'] = ~kwargs.get('transpose', False)
    buf, xe, xedges, yw, yedges = get_hist2d_data_eq(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )

    extent = [ yedges[0], yedges[-1], xedges[0], xedges[-1] ]
    kwargs.setdefault( 'origin', 'lower' )
    kwargs.setdefault( 'extent', extent )

    res = P.imshow(buf, *args, **kwargs)

    return colorbar_or_not(res, colorbar)

def matshow(output, *args, **kwargs):
    """Plot matrix using pyplot.matshow"""
    ifNd(output, 2)

    mask = kwargs.pop( 'mask', None )
    colorbar = kwargs.pop( 'colorbar', None )
    kwargs.setdefault( 'fignum', False )

    buf = get_2d_buffer(output, transpose=kwargs.pop('transpose', False), mask=mask)

    res = P.matshow(buf, **kwargs)

    return colorbar_or_not(res, colorbar)

def surface_hist2(output, *args, **kwargs):
    Z, xedges, yedges = get_hist2d_data(output, kwargs)

    xc=(xedges[1:]+xedges[:-1])*0.5
    yc=(yedges[1:]+yedges[:-1])

    X, Y = N.meshgrid(xc, yc, indexing='ij')

    colorbar = kwargs.pop('colorbar', False)

    ax = P.gca()
    res = ax.plot_surface(X, Y, Z, *args, **kwargs)

    return colorbar_or_not_3d(res, colorbar)

def apply_colors(buf, kwargs, colorsname):
    cmap = kwargs.pop('cmap', False)
    if cmap==True:
        cmap='viridis'

    if not cmap:
        return None, None

    import matplotlib.colors as colors
    from matplotlib import cm

    bmin, bmax = buf.min(), buf.max()
    norm = (buf-bmin)/(bmax-bmin)

    cmap = cm.get_cmap(cmap)
    res = cmap(norm)
    kwargs[colorsname] = res
    return res, cmap

def wireframe_hist2(output, *args, **kwargs):
    Z, xedges, yedges = get_hist2d_data(output, kwargs)

    xc=(xedges[1:]+xedges[:-1])*0.5
    yc=(yedges[1:]+yedges[:-1])

    X, Y = N.meshgrid(xc, yc, indexing='ij')

    colors, cmap = apply_colors(Z, kwargs, 'facecolors')
    colorbar = kwargs.pop('colorbar', False)

    ax = P.gca()
    if colors is not None:
        kwargs['rcount']=Z.shape[0]
        kwargs['ccount']=Z.shape[1]
        kwargs['shade']=False
        res = ax.plot_surface(X, Y, Z, **kwargs)
        res.set_facecolor((0,0,0,0))

        return colorbar_or_not_3d(res, colorbar, Z, cmap=cmap)

    res = ax.plot_wireframe(X, Y, Z, *args, **kwargs)
    return res

def bar3d_hist2(output, *args, **kwargs):
    Zw, xedges, yedges = get_hist2d_data(output, kwargs)

    xw=xedges[1:]-xedges[:-1]
    yw=yedges[1:]-yedges[:-1]

    X, Y = N.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    X, Y = X.ravel(), Y.ravel()

    Xw, Yw = N.meshgrid(xw, yw, indexing='ij')
    Xw, Yw, Zw = Xw.ravel(), Yw.ravel(), Zw.ravel()
    Z = N.zeros_like(Zw)

    colors, cmap = apply_colors(Zw, kwargs, 'color')
    colorbar = kwargs.pop('colorbar', False)

    # if colorizetop:
        # nel, ncol = colors.shape
        # newcolors = N.ones( (nel, 6, ncol), dtype=colors.dtype )
        # newcolors[:,1,:]=colors
        # newcolors.shape=(nel*6, ncol)
        # kwargs['color']=newcolors

    ax = P.gca()
    res = ax.bar3d(X, Y, Z, Xw, Yw, Zw, *args, **kwargs)

    return colorbar_or_not_3d(res, colorbar, Zw, cmap=cmap)


# def graph_plot( g, *args, **kwargs ):
    # """Plot TGraph using pyplot.plot"""
    # x, y = R2N.get_buffers_graph( g )
    # return P.plot( x, y, *args, **kwargs )

# def errorbar_graph( g, *args, **kwargs ):
    # """Plot TGraphErrors using pyplot.errorbar"""
    # x, y = R2N.get_buffers_graph( g )
    # ex, ey = R2N.get_err_buffers_graph( g )
    # if ( ex==0.0 ).all(): ex = None
    # if ( ey==0.0 ).all(): ey = None
    # return P.errorbar( x, y, ey, ex, *args, **kwargs )
# ##end def

# def errorbar_graph_asymm( g, *args, **kwargs ):
    # """Plot TGraphErrors using pyplot.errorbar"""
    # x, y = R2N.get_buffers_graph( g )
    # exl, exh, eyl, eyh = R2N.get_err_buffers_graph_asymm( g )
    # ex = N.array( ( exl, exh ) )
    # ey = N.array( ( eyl, eyh ) )
    # if ( ex==0.0 ).all(): ex = None
    # if ( ey==0.0 ).all(): ey = None
    # return P.errorbar( x, y, ey, ex, *args, **kwargs )

# def spline_plot( spline, *args, **kwargs ):
    # """Plot TSpline using pyplot.plot"""
    # xmin = kwargs.pop( 'xmin', spline.GetXmin() )
    # xmax = kwargs.pop( 'xmax', spline.GetXmax() )
    # n    = kwargs.pop( 'n', spline.GetNp() )

    # x = N.linspace( xmin, xmax, n )
    # fcn = N.frompyfunc( spline.Eval, 1, 1 )
    # y = fcn( x )

    # return P.plot( x, y, *args, **kwargs )

def bind():
    setattr( R.SingleOutput, 'plot',      plot_points )
    setattr( R.SingleOutput, 'plot_vs',   plot_vs_points )
    setattr( R.SingleOutput, 'vs_plot',   vs_plot_points )
    setattr( R.SingleOutput, 'plot_bar',  bar_hist1 )
    setattr( R.SingleOutput, 'plot_hist', plot_hist1 )
    setattr( R.SingleOutput, 'plot_matshow', matshow )
    # setattr( R.TH1, 'errorbar', errorbar_hist1 )

    setattr( R.SingleOutput, 'plot_pcolorfast', pcolorfast_hist2 )
    setattr( R.SingleOutput, 'plot_pcolormesh', pcolormesh_hist2 )
    setattr( R.SingleOutput, 'plot_pcolor',     pcolor_hist2 )
    setattr( R.SingleOutput, 'plot_imshow',     imshow_hist2 )

    setattr( R.SingleOutput, 'plot_bar3d',      bar3d_hist2 )
    setattr( R.SingleOutput, 'plot_surface',    surface_hist2 )
    setattr( R.SingleOutput, 'plot_wireframe',  wireframe_hist2 )

    # setattr( R.TGraph,            'plot',     graph_plot )
    # setattr( R.TGraphErrors,      'errorbar', errorbar_graph )
    # setattr( R.TGraphAsymmErrors, 'errorbar', errorbar_graph_asymm )

    # setattr( R.TSpline, 'plot', spline_plot )

