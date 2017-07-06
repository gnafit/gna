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
    xsize = 501;
    ysize = 501;
    x = N.linspace( -0., 6., xsize, dtype='d' )
    y = N.linspace( -2., 4., ysize, dtype='d' )
    X, Y = N.meshgrid( x, y, indexing='ij' )
    Z = X**2 + Y**2
    Z1 = 2.0*(X-3.0)*(X-3.0) + 5.0*(Y-1.0)*(Y-1.0) + 3.0*(X-3.0)*(Y-1.0)
    Z2 = 0.05*X*X*X + 0.1*Y*Y*Y + Z1

    p = R.Paraboloid(Z1.shape[0], Z1.shape[1], numpy_to_eigen( Z1 ), 1, 3., 0.5 )
    masks = []
    for level in opts.levels:
	mat = R.Eigen.MatrixXd( xsize, ysize )
        p.GetCrossSectionExtendedAutoDev( mat, level )
        csc = eigen_to_numpy( mat, 'bool' ).copy()
        masks.append( csc )

        Zl=N.ma.array(Z, mask=~csc)
        Zvalues = Zl.compressed()
        if Zvalues.shape[0] != 0:
            print( 'Level %f (min/max): '%level, Zvalues.min(), Zvalues.max() )

    plotl( X, Y, Z1, levels=opts.levels, masks=masks )
    plt.contour( X, Y, Z2, levels=opts.levels)

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
