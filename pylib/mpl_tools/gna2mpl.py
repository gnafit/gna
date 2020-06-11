#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
import numpy as N
from . import root2numpy as R2N
from mpl_tools import helpers
from matplotlib import pyplot as P
from gna.bindings import DataType, provided_precisions

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

def ifSameType(output1, output2):
    dtype1, dtype2 = output1.datatype(), output2.datatype()

    if dtype1!=dtype2:
        raise Exception('Outputs are not consistent')

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

    Plotter = kwargs.pop('axis', P)

    return Plotter.plot(points, *args, **kwargs )

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

    Plotter = kwargs.pop('axis', P)

    return Plotter.plot(pointsx, pointsy, *args, **kwargs )

def vs_plot_points(outputx, outputy, *args, **kwargs):
    return plot_vs_points(outputy, outputx, *args, **kwargs)

def get_1d_data(output, scale=None):
    ifNd(output, 1)

    buf = output.single().data().copy()

    dtype=output.datatype()
    if dtype.kind==2:
        lims = N.array(dtype.edges)
        width = lims[1:] - lims[:-1]
    else:
        lims = N.arange(dtype.shape[0]+1, dtype='d')
        width = 1.0

    if scale is None:
        pass
    elif scale=='width':
        buf/=width
    elif isinstance(scale, (float, int)):
        buf*=float(scale)
    else:
        raise Exception('Unsupported scale:', scale)

    return buf, lims, width

def plot_hist(output, *args, **kwargs):
    """Plot 1-dimensinal output using pyplot.plot

    executes pyplot.plot(x, y, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    Options:
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.plot() result
    """
    scale = kwargs.pop('scale', None)
    height, lims, _ = get_1d_data(output, scale=scale)

    diff=kwargs.pop('diff', None)
    if diff is not None:
        ifSameType(output, diff)
        height1, lims1, _ = get_1d_data(diff, scale)

        height-=height1

    ratio=kwargs.pop('ratio', None)
    if ratio is not None:
        ifSameType(output, ratio)
        height1, lims1, _ = get_1d_data(ratio, scale)

        height/=height1
        height[N.isnan(height)]=1.0

    offset_ratio=kwargs.pop('offset_ratio', None)
    if offset_ratio is not None:
        ifSameType(output, offset_ratio)
        height1, lims1, _ = get_1d_data(offset_ratio, scale)

        height/=height1
        height-=1.0
        height[N.isnan(height)]=0.0

    return helpers.plot_hist(lims, height, *args, **kwargs)

def plot_hist_centers(output, *args, **kwargs):
    """Plot 1-dimensinal output using pyplot.plot

    executes pyplot.plot(x, y, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    Options:
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.plot() result
    """
    height, lims, _ = get_1d_data(output, scale=kwargs.pop('scale', None))
    centers = (lims[1:] + lims[:-1])*0.5

    Plotter = kwargs.pop('axis', P)

    return Plotter.plot(centers, height, *args, **kwargs )

def bar_hist( output, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.bar

    executes pyplot.bar(left, height, width, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    Options:
        divide=N - divide bin width by N
        shift=N  - shift bin left edge by N*width
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.bar() result
    """
    divide = kwargs.pop( 'divide', None )
    shift  = kwargs.pop( 'shift', 0 )

    height, lims, width = get_1d_data(output, scale=kwargs.pop('scale', None))
    left  = lims[:-1]

    if divide:
        width/=divide
        left+=width*shift

    kwargs.setdefault( 'align', 'edge' )
    Plotter = kwargs.pop('axis', P)

    return Plotter.bar( left, height, width, *args, **kwargs )

def errorbar_hist(output, yerr=None, *args, **kwargs):
    """Plot 1-dimensinal histogram using pyplot.errorbar

    executes pyplot.errorbar(x, y, yerr, xerr, *args, **kwargs) with x, y and xerr overridden
    all other arguments passes as is.

    Options:
        yerr=array or 'stat' - Y errors
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.errorbar() result
    """
    Y, lims, width = get_1d_data(output)

    return helpers.plot_hist_errorbar(lims, Y, yerr, *args, **kwargs )

def get_2d_buffer(output, transpose=False, mask=None):
    if isinstance(output, N.ndarray):
        buf = output
    else:
        buf = output.data().copy()

    if mask is not None:
        buf = N.ma.array(buf, mask=buf==mask)

    if transpose:
        buf = buf.T

    return buf

def get_2d_data(output, kwargs):
    ifNd(output, 2)

    mask      = kwargs.pop( 'mask', None )
    transpose = kwargs.pop( 'transpose', False )

    dtype=output.datatype()

    if dtype.kind == 2:
        xedges, yedges = N.array(dtype.edgesNd[0]), N.array(dtype.edgesNd[1])
    else:
        xedges = N.arange(dtype.shape[0]+1, dtype='d')
        yedges = N.arange(dtype.shape[1]+1, dtype='d')

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

def get_2d_data_eq(output, kwargs):
    buf, xedges, yedges = get_2d_data(output, kwargs)

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

def pcolorfast(output, *args, **kwargs):
    kwargs['transpose'] = ~kwargs.get('transpose', False)
    buf, xe, xedges, yw, yedges = get_2d_data_eq(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )
    x = [yedges[0], yedges[-1]]
    y = [xedges[0], xedges[-1]]

    ax = P.gca()
    res = ax.pcolorfast( x, y, buf, *args, **kwargs )

    return colorbar_or_not(res, colorbar)

def pcolormesh(output, *args, **kwargs):
    buf, xedges, yedges = get_2d_data(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )

    x, y = N.meshgrid(xedges, yedges, indexing='ij')

    res = P.pcolormesh(x, y, buf, *args, **kwargs)

    return colorbar_or_not(res, colorbar)

def pcolor(output, *args, **kwargs):
    buf, xedges, yedges = get_2d_data(output, kwargs)
    colorbar  = kwargs.pop( 'colorbar', None )

    x, y = N.meshgrid(xedges, yedges, indexing='ij')

    res = P.pcolor(x, y, buf, *args, **kwargs)

    return colorbar_or_not(res, colorbar)

def imshow(output, *args, **kwargs):
    kwargs['transpose'] = ~kwargs.get('transpose', False)
    buf, xe, xedges, yw, yedges = get_2d_data_eq(output, kwargs)
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

def surface(output, *args, **kwargs):
    Z, xedges, yedges = get_2d_data(output, kwargs)

    xc=(xedges[1:]+xedges[:-1])*0.5
    yc=(yedges[1:]+yedges[:-1])*0.5

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

def wireframe(output, *args, **kwargs):
    Z, xedges, yedges = get_2d_data(output, kwargs)

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
        res.set_facecolor((0, 0, 0, 0))

        return colorbar_or_not_3d(res, colorbar, Z, cmap=cmap)

    res = ax.plot_wireframe(X, Y, Z, *args, **kwargs)
    return res

def wireframe_points_vs(output, xmesh, ymesh, *args, **kwargs):
    transpose=kwargs.pop('transpose', False)
    Z = get_2d_buffer(output, transpose=transpose, mask=kwargs.pop('mask', None))

    X, Y = xmesh, ymesh
    if transpose:
        X, Y = Y.T, X.T

    colors, cmap = apply_colors(Z, kwargs, 'facecolors')
    colorbar = kwargs.pop('colorbar', False)

    ax = P.gca()
    if colors is not None:
        kwargs['rcount']=Z.shape[0]
        kwargs['ccount']=Z.shape[1]
        kwargs['shade']=False
        res = ax.plot_surface(X, Y, Z, **kwargs)
        res.set_facecolor((0, 0, 0, 0))

        return colorbar_or_not_3d(res, colorbar, Z, cmap=cmap)

    res = ax.plot_wireframe(X, Y, Z, *args, **kwargs)
    return res

def bar3d(output, *args, **kwargs):
    Zw, xedges, yedges = get_2d_data(output, kwargs)

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

def bind():
    for p in provided_precisions:
        setattr( R.SingleOutputT(p), 'plot',              plot_points )
        setattr( R.SingleOutputT(p), 'plot_vs',           plot_vs_points )
        setattr( R.SingleOutputT(p), 'vs_plot',           vs_plot_points )
        setattr( R.SingleOutputT(p), 'plot_bar',          bar_hist )
        setattr( R.SingleOutputT(p), 'plot_hist',         plot_hist )
        setattr( R.SingleOutputT(p), 'plot_hist_centers', plot_hist_centers )
        setattr( R.SingleOutputT(p), 'plot_errorbar',     errorbar_hist )
        setattr( R.SingleOutputT(p), 'plot_matshow',      matshow )

        setattr( R.SingleOutputT(p), 'plot_pcolorfast',   pcolorfast )
        setattr( R.SingleOutputT(p), 'plot_pcolormesh',   pcolormesh )
        setattr( R.SingleOutputT(p), 'plot_pcolor',       pcolor )
        setattr( R.SingleOutputT(p), 'plot_imshow',       imshow )

        setattr( R.SingleOutputT(p), 'plot_bar3d',        bar3d )
        setattr( R.SingleOutputT(p), 'plot_surface',      surface )
        setattr( R.SingleOutputT(p), 'plot_wireframe',    wireframe )

        setattr( R.SingleOutputT(p), 'plot_wireframe_vs', wireframe_points_vs )
