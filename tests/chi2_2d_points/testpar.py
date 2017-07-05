#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import itertools as I
from matplotlib import pyplot as plt
R.GNAObject

def plotl( X, Y, Z, **kwargs ):
    levels = kwargs.pop( 'levels', [] )
    masks  = kwargs.pop( 'masks', [] )
    assert len(levels)==len(masks)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( kwargs.pop( 'xlabel', 'axis x' ) )
    ax.set_ylabel( kwargs.pop('ylabel', 'axis y') )
    ax.set_title( kwargs.pop('title', 'Paraboloid') )

    ax.contour( X, Y, Z, levels )

    markers = 'ov^s'
    for level, mask, marker in zip(levels, masks, I.cycle( markers ) ):
        ax.plot( X[mask], Y[mask], marker, label='L=%g'%level )

    ax.legend( loc='upper right' )

def eigen_to_numpy( eigen, dtype ):
    return N.frombuffer( eigen.data(), dtype='d', count=eigen.size() ).reshape( eigen.rows(), eigen.cols(), order='F' ).astype(dtype)

def numpy_to_eigen( np ):
    return np.ravel( order='F' )

def main( opts ):
    x = N.linspace( -10., 10., 201, dtype='d' )
    y = N.linspace( -10., 10., 201, dtype='d' )
    X, Y = N.meshgrid( x, y, indexing='ij' )
    Z = X**2 + Y**2

    p = R.Paraboloid( Z.shape[0], Z.shape[1], numpy_to_eigen( Z ), 1, 1.0 )

    masks = []
    for level in opts.levels:
        csc = p.GetCrossSectionExtendedAutoDev( level )
        csc = eigen_to_numpy( csc, 'bool' ).copy()
        masks.append( csc )

        Zl=N.ma.array(Z, mask=~csc)
        Zvalues = Zl.compressed()
        print( 'Level %f (min/max): '%level, Zvalues.min(), Zvalues.max() )

    plotl( X, Y, Z, levels=opts.levels, masks=masks )

    if opts.output:
        plt.savefig( opts.output )
        print( 'Save figure to:', opts.output )

    if opts.show:
        plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument( '-s', '--show', action='store_true', help='show figures' )
    parser.add_argument( '-l', '--levels', nargs='+', default=[1.0, 4.0, 9.0], type=float, help='levels to process')
    parser.add_argument( '-o', '--output', help='figure output' )
    main( parser.parse_args() )
