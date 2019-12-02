#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N

def plot_hist( lims, height, *args, **kwargs ):
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

def add_colorbar( colormapable, **kwargs ):
    """Add a colorbar to the axis with height aligned to the axis"""
    rasterized = kwargs.pop( 'rasterized', True )
    minorticks = kwargs.pop( 'minorticks', False )
    minorticks_values = kwargs.pop( 'minorticks_values', None )

    ax = P.gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = P.gcf().colorbar( colormapable, cax=cax, **kwargs )

    if minorticks:
        if type(minorticks) is str:
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
    P.sca( ax )
    return cbar

def savefig( name, *args, **kwargs ):
    """Save fig and print output filename"""
    if not name: return
    if type(name)==list:
        for n in name:
            savefig( n, *args, **kwargs.copy() )
        return

    suffix = kwargs.pop( 'suffix', None )
    if not isinstance(suffix, str):
        suffix = '_'+ '_'.join(suffix)

    addext = kwargs.pop( 'addext', [] )
    if suffix:
        from os.path import splitext
        basename, ext = splitext( name )
        name = basename+suffix+ext

    P.savefig( name, *args, **kwargs )
    print( 'Save figure', name )

    if addext:
        if not type( addext )==list:
            addext = [ addext ]
        from os import path
        basename, extname = path.splitext( name )
        for ext in addext:
            name = '%s.%s'%( basename, ext )
            print( 'Save figure', name )
            P.savefig( name, *args, **kwargs )

def add_to_labeled_items(o, l, ax=None):
    ax = ax or P.gca()
    ocurrent, lcurrent = ax.get_legend_handles_labels()
    ocurrent.append( o )
    lcurrent.append( l )
