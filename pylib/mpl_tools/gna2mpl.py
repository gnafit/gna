#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from . import root2numpy as R2N
from mpl_tools import helpers
from mpl_tools.helpers import _colorbar_or_not, _colorbar_or_not_3d
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

    index = kwargs.pop("index", False)
    if index:
        lims = N.array([i for i, _ in enumerate(lims)])

    return Plotter.plot(points, *args, **kwargs )

def plot_vs_points(outputy, outputx, *args, **kwargs):
    """Plot array using pyplot.plot

    executes pyplot.plot(y, *args, **kwargs) with first argument overridden,
    all other arguments are passed as is.

    returns pyplot.plot() result
    """
    fcn = kwargs.pop('fcn', None)
    pointsx = get_array(outputx)
    pointsy = get_array(outputy)

    if kwargs.pop('transpose', False):
        pointsx, pointsy=pointsx.T, pointsy.T

    if fcn:
        pointsx, pointsy = fcn(pointsx, pointsy)

    ratio=kwargs.pop('ratio', None)
    logratio=kwargs.pop('logratio', None)
    if ratio is not None:
        outputy1 = ratio
    elif logratio is not None:
        outputy1 = logratio
    else:
        outputy1 = None

    if outputy1 is not None:
        ifSameType(outputy, outputy1)
        pointsy1 = get_array(outputy1)
        if fcn:
            _, pointsy1 = fcn(pointsx, pointsy1)
        mask = (pointsy==0.0)*(pointsy1==0.0)
        pointsy/=pointsy1
        pointsy[mask]=1.0
        if logratio is not None:
            pointsy=N.log(pointsy)

    Plotter = kwargs.pop('axis', P)
    ravel = kwargs.pop('ravel', False)
    if ravel:
        pointsx = pointsx.ravel()
        pointsy = pointsy.ravel()

        asort = N.argsort(pointsx)
        pointsx=pointsx[asort]
        pointsy=pointsy[asort]

    return Plotter.plot(pointsx, pointsy, *args, **kwargs )

def vs_plot_points(outputx, outputy, *args, **kwargs):
    return plot_vs_points(outputy, outputx, *args, **kwargs)

def get_array(output):
    if isinstance(output, (N.ndarray, list)):
        return N.asanyarray(output)

    return output.data().copy()

def get_1d_data(output, scale=None, allow_diagonal=False, sqrt=False):
    dtype = output.datatype()
    ndim = dtype.shape.size()

    if ndim==1:
        buf = output.single().data().copy()
    elif ndim==2 and allow_diagonal:
        buf = output.single().data().diagonal().copy()
    else:
        raise Exception('Output is supposed to be {}: '.format(allow_diagonal and '1d/2d' or '1d')+str(dtype))

    if ndim==1 and dtype.kind==2:
        lims = N.array(dtype.edges)
        width = lims[1:] - lims[:-1]
    else:
        lims = N.arange(buf.size+1, dtype='d')
        width = 1.0

    if scale is None:
        pass
    elif scale=='width':
        buf/=width
    elif isinstance(scale, (float, int)):
        buf*=float(scale)
    else:
        raise Exception('Unsupported scale:', scale)

    if sqrt:
        buf = buf**0.5

    return buf, lims, width

def plot_hist(output, *args, allow_diagonal=False, **kwargs):
    """Plot 1-dimensinal output using pyplot.plot

    executes pyplot.plot(x, y, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    Options:
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.plot() result
    """
    scale = kwargs.pop('scale', None)
    sqrt = kwargs.pop('sqrt', False)
    height, lims, _ = get_1d_data(output, scale=scale, allow_diagonal=allow_diagonal, sqrt=sqrt)

    diff=kwargs.pop('diff', None)
    if diff is not None:
        ifSameType(output, diff)
        height1, _, _ = get_1d_data(diff, scale, allow_diagonal=allow_diagonal, sqrt=sqrt)

        height-=height1

    ratio=kwargs.pop('ratio', None)
    logratio=kwargs.pop('logratio', None)
    if ratio is not None:
        ifSameType(output, ratio)
        height1, _, _ = get_1d_data(ratio, scale, allow_diagonal=allow_diagonal, sqrt=sqrt)
        mask = (height==0.0)*(height1==0.0)
        height/=height1
        height[mask]=1.0
    elif logratio is not None:
        ifSameType(output, logratio)
        height1, _, _ = get_1d_data(logratio, scale, allow_diagonal=allow_diagonal, sqrt=sqrt)
        mask = (height==0.0)*(height1==0.0)
        height/=height1
        height[mask]=1.0
        height=N.log(height)

    index = kwargs.pop("index", False)
    if index:
        lims = N.array([i for i, _ in enumerate(lims)])

    return helpers.plot_hist(lims, height, *args, **kwargs)

def plot_hist_centers(output, *args, allow_diagonal=False, **kwargs):
    """Plot 1-dimensinal output using pyplot.plot

    executes pyplot.plot(x, y, *args, **kwargs) with first two arguments overridden
    all other arguments are passed as is.

    Options:
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.plot() result
    """
    sqrt = kwargs.pop('sqrt', False)

    height, lims, _ = get_1d_data(output, scale=kwargs.pop('scale', None), allow_diagonal=allow_diagonal, sqrt=sqrt)
    centers = (lims[1:] + lims[:-1])*0.5

    Plotter = kwargs.pop('axis', P)

    index = kwargs.pop("index", False)
    if index:
        lims = N.array([i for i, _ in enumerate(lims)])

    return Plotter.plot(centers, height, *args, **kwargs )

def bar_hist( output, *args, allow_diagonal=False, **kwargs ):
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
    sqrt = kwargs.pop('sqrt', False)

    height, lims, width = get_1d_data(output, scale=kwargs.pop('scale', None), allow_diagonal=allow_diagonal, sqrt=sqrt)
    left  = lims[:-1]

    if divide:
        width/=divide
        left+=width*shift

    kwargs.setdefault( 'align', 'edge' )
    Plotter = kwargs.pop('axis', P)

    index = kwargs.pop("index", False)
    if index:
        left = N.array([i for i, _ in enumerate(left)])
        width = N.ones(left.shape)

    return Plotter.bar( left, height, width, *args, **kwargs )

def errorbar_hist(output, yerr=None, *args, allow_diagonal=False, **kwargs):
    """Plot 1-dimensinal histogram using pyplot.errorbar

    executes pyplot.errorbar(x, y, yerr, xerr, *args, **kwargs) with x, y and xerr overridden
    all other arguments passes as is.

    Options:
        yerr=array or 'stat' - Y errors
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.errorbar() result
    """
    sqrt = kwargs.pop('sqrt', False)

    Y, lims, _ = get_1d_data(output, allow_diagonal=allow_diagonal, sqrt=sqrt)

    index = kwargs.pop("index", False)
    if index:
        lims = N.array([i for i, _ in enumerate(lims)])

    return helpers.plot_hist_errorbar(lims, Y, yerr, *args, **kwargs)

def get_2d_buffer(output, transpose=False, mask=None, preprocess=None):
    if isinstance(output, N.ndarray):
        buf = output
    else:
        buf = output.data().copy()

    if mask is not None:
        buf = N.ma.array(buf, mask=buf==mask)

    if transpose:
        buf = buf.T

    if preprocess:
        buf = preprocess(buf)

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

def pcolorfast(output, *args, **kwargs):
    kwargs['transpose'] = ~kwargs.get('transpose', False)
    buf, xe, xedges, yw, yedges = get_2d_data_eq(output, kwargs)
    x = [yedges[0], yedges[-1]]
    y = [xedges[0], xedges[-1]]
    return helpers.pcolorfast(x, y, buf, *args, **kwargs)

def pcolormesh(output, *args, **kwargs):
    buf, xedges, yedges = get_2d_data(output, kwargs)
    x, y = N.meshgrid(xedges, yedges, indexing='ij')
    return helpers.pcolormesh(x, y, buf, *args, **kwargs)

def pcolor(output, *args, **kwargs):
    buf, xedges, yedges = get_2d_data(output, kwargs)
    x, y = N.meshgrid(xedges, yedges, indexing='ij')
    return helpers.pcolor(x, y, buf, *args, **kwargs)

def imshow(output, *args, **kwargs):
    kwargs['transpose'] = ~kwargs.get('transpose', False)
    buf, xe, xedges, yw, yedges = get_2d_data_eq(output, kwargs)
    extent = [ yedges[0], yedges[-1], xedges[0], xedges[-1] ]
    kwargs.setdefault( 'origin', 'lower' )
    kwargs.setdefault( 'extent', extent )
    return helpers.imshow(buf, *args, **kwargs)

def matshow(output, *args, **kwargs):
    """Plot matrix using pyplot.matshow"""
    ifNd(output, 2)
    mask = kwargs.pop( 'mask', None )
    preprocess = kwargs.pop( 'preprocess', None )
    kwargs.setdefault( 'fignum', False )
    buf = get_2d_buffer(output, transpose=kwargs.pop('transpose', False), mask=mask, preprocess=preprocess)
    return helpers.matshow(buf, **kwargs)

def surface(output, *args, **kwargs):
    Z, xedges, yedges = get_2d_data(output, kwargs)
    xc=(xedges[1:]+xedges[:-1])*0.5
    yc=(yedges[1:]+yedges[:-1])*0.5
    X, Y = N.meshgrid(xc, yc, indexing='ij')
    return helpers.plot_surface(X, Y, Z, *args, **kwargs)

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

        return _colorbar_or_not_3d(res, colorbar, Z, cmap=cmap)

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

        return _colorbar_or_not_3d(res, colorbar, Z, cmap=cmap)

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

    return _colorbar_or_not_3d(res, colorbar, Zw, cmap=cmap)

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
