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

    return P.plot( x, y, *args, **kwargs )

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
