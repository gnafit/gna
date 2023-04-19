#!/usr/bin/env python

from matplotlib import pyplot as P
import numpy as N
from collections.abc import Mapping

def plot_hist(lims, height, *args, **kwargs):
    """Plot histogram with lines. Like bar(), but without lines between bars."""
    zero_value = kwargs.pop( 'zero_value', 0.0 )

    y = N.empty( len(height)*2+2 )
    y[0], y[-1]=zero_value, zero_value
    y[1:-1] = N.vstack( ( height, height ) ).ravel( order='F' )
    x = N.vstack( ( lims, lims ) ).ravel( order='F' )

    if kwargs.pop( 'noedge', False ):
        x = x[1:-1]
        y = y[1:-1]

    Plotter = kwargs.pop('axis', P)

    return Plotter.plot( x, y, *args, **kwargs )

def plot_hist_errorbar(lims, Y, yerr=None, *args, **kwargs):
    """Plot 1-dimensinal histogram using pyplot.errorbar

    executes pyplot.errorbar(x, y, yerr, xerr, *args, **kwargs) with x, y and xerr overridden
    all other arguments passes as is.

    Options:
        yerr=array or 'stat' - Y errors
        scale=float or 'width' - multiply bin by a scale or divide by bin width

    returns pyplot.errorbar() result
    """
    scale = kwargs.pop('scale', None)

    width = lims[1:]-lims[:-1]
    X     = (lims[1:]+lims[:-1])*0.5
    Xerr=width*0.5

    if isinstance(yerr, str) and yerr=='stat':
        Yerr=Y**0.5
    else:
        Yerr=yerr

    if scale is None or Yerr is None:
        pass
    elif scale=='width':
        Yerr=Yerr/width
        Y=Y/width
    else:
        Yerr=Yerr*scale
        Y=Y*scale

    kwargs.setdefault('fmt', 'none')

    Plotter = kwargs.pop('axis', P)

    return Plotter.errorbar(X, Y, Yerr, Xerr, *args, **kwargs)

def plot_bar( lims, height, *args, **kwargs ):
    """Plot bars with edges specified"""
    kwargs.setdefault( 'align', 'edge' )
    pack = kwargs.pop( 'pack', None )

    lims = N.asanyarray( lims )
    left = lims[:-1]
    widths  = lims[1:] - left

    if pack is not None:
        i, n = pack
        widths/=float(n)
        left+=i*widths

    Plotter = kwargs.pop('axis', P)

    return Plotter.bar(left, height, widths, *args, **kwargs )

def savefig(name, *args, **kwargs):
    """Save fig and print output filename"""
    close = kwargs.pop('close', False)
    separator = kwargs.pop('sep', False)
    filenames = ()
    if name:
        if isinstance(name, list):
            for n in name:
                filenames+=savefig( n, *args, **kwargs.copy() )
        else:
            suffix = kwargs.pop( 'suffix', None )
            addext = kwargs.pop( 'addext', [] )
            if suffix:
                from os.path import splitext
                basename, ext = splitext( name )

                if not isinstance(suffix, str):
                    suffix = '_'.join(suffix)

                name = basename+suffix+ext

            P.savefig( name, *args, **kwargs )
            print( 'Save figure', name )
            filenames+=name,

            if addext:
                if not isinstance(addext, list):
                    addext = [ addext ]
                from os import path
                basename, extname = path.splitext( name )
                for ext in addext:
                    name = '%s.%s'%( basename, ext )
                    P.savefig( name, *args, **kwargs )
                    print( 'Save figure', name )
                    filenames+=name,
    if close:
        P.close()

    return filenames

def add_colorbar( colormapable, **kwargs ):
    """Add a colorbar to the axis with height aligned to the axis"""
    rasterized = kwargs.pop( 'rasterized', True )
    minorticks = kwargs.pop( 'minorticks', False )
    label = kwargs.pop( 'label', None )
    minorticks_values = kwargs.pop( 'minorticks_values', None )

    ax = P.gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = P.gcf().colorbar( colormapable, cax=cax, **kwargs )

    if minorticks:
        if isinstance(minorticks, str):
            if minorticks=='linear':
                pass
            elif minorticks=='log':
                minorticks_values = colormapable.norm( minorticks_values )

            l1, l2 = cax.get_ylim()
            minorticks_values = minorticks_values[ (minorticks_values>=l1)*(minorticks_values<=l2) ]
            cax.yaxis.set_ticks(minorticks_values, minor=True)
        else:
            cax.minorticks_on()

    if rasterized:
        cbar.solids.set_rasterized( True )

    if label is not None:
        cbar.set_label(label, rotation=270)
    P.sca( ax )
    return cbar

def add_colorbar_3d(res, cbaropt={}, mappable=None, cmap=None):
    """Add a colorbar to the 3d axis with height aligned to the axis"""
    cbaropt.setdefault('aspect', 4)
    cbaropt.setdefault('shrink', 0.5)

    if mappable is None:
        cbar = P.colorbar(res, **cbaropt)
    else:
        colourMap = P.cm.ScalarMappable()
        colourMap.set_array(mappable)
        cbar = P.colorbar(colourMap, **cbaropt)

    return res, cbar

def _colorbar_or_not(res, cbaropt):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, Mapping):
        cbaropt = {}

    cbar = add_colorbar(res, **cbaropt)

    return res, cbar

def _colorbar_or_not_3d(res, cbaropt, mappable=None, cmap=None):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, Mapping):
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

def add_to_labeled_items(o, l, ax=None):
    ax = ax or P.gca()
    ocurrent, lcurrent = ax.get_legend_handles_labels()
    ocurrent.append( o )
    lcurrent.append( l )

def _patch_with_colorbar(fcn, mode3d=False):
    '''Patch pyplot.function or ax.method by adding a "colorbar" option'''
    returner = mode3d and _colorbar_or_not_3d or _colorbar_or_not
    if isinstance(fcn, str):
        def newfcn(*args, **kwargs):
            colorbar=kwargs.pop('colorbar', None)
            ax = P.gca()
            actual_fcn = getattr(ax, fcn)
            res = actual_fcn(*args, **kwargs)
            return returner(res, colorbar)
    else:
        def newfcn(*args, **kwargs):
            colorbar=kwargs.pop('colorbar', None)
            res = fcn(*args, **kwargs)
            return returner(res, colorbar)

    return newfcn


pcolorfast = _patch_with_colorbar('pcolorfast')
pcolor     = _patch_with_colorbar(P.pcolor)
pcolormesh = _patch_with_colorbar(P.pcolormesh)
imshow     = _patch_with_colorbar(P.imshow)
matshow    = _patch_with_colorbar(P.matshow)
plot_surface = _patch_with_colorbar('plot_surface', mode3d=True)
