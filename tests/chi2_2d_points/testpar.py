#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N 
import itertools as I
from matplotlib import pyplot as plt
R.GNAObject

def plotl( X, Y, Z, **kwargs ):
    print(X.shape)
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

def points_to_mask( points, xsize, ysize ):
    x = N.zeros((xsize, ysize), 'bool')
    x[points[0], points[1]] = 1;
    print(x)
    return x

def mask_to_points( mask ):
    return N.transpose(N.nonzero(mask))

def invert_values( x ):
    return True if x==False else False

invert_ = N.vectorize(invert_values)

def main( opts ):
    xsize = 501;
    ysize = 501;
    ppp = N.matrix([[1, 3, 5], [2, 5, 2]])
    print(mask_to_points(points_to_mask(ppp, 6, 6)))
    x = N.linspace( -0., 6., xsize, dtype='d' )
    y = N.linspace( -2., 4., ysize, dtype='d' )
    X, Y = N.meshgrid( x, y, indexing='ij' )
    Z = X**2 + Y**2
    Z1 = 2.0*(X-3.0)*(X-3.0) + 5.0*(Y-1.0)*(Y-1.0) + 3.0*(X-3.0)*(Y-1.0)
    Z2 = 0.05*X*X*X + 0.1*Y*Y*Y + Z1
    x = N.zeros((10, 10), 'bool')
    x[::3, ::3] = 1
    x=invert_(x)
    print(x)
    p = R.Paraboloid(Z1.shape[0], Z1.shape[1], numpy_to_eigen( Z1 ), 6. / xsize, 6. / ysize, 1, 1., 0.1 )
    masks = []
    #invert_values = N.vectorize(invert_values)
    for level in opts.levels:
	mat = R.Eigen.MatrixXd( xsize, ysize )
        p.GetCrossSectionExtendedAutoDev( mat, level )
        csc = eigen_to_numpy( mat, 'bool' ).copy()
        #map(invert_values, csc)
        print(csc)
        print(csc.shape)
        csc1 = invert_( csc )
        print (csc1)
        print(csc1.shape)
        xslice = N.arange(0, xsize, 1)
        xslice = N.delete(xslice, N.arange(0, xsize, 100))
        #print(xslice)
        yslice = N.arange(ysize)
        #csc = csc.array()*
        #csc[xslice, xslice] = 0
        #csc = invert_values(csc)
        masks.append( csc1 )
        Zl=N.ma.array(Z, mask=~csc1)
        Zvalues = Zl.compressed()
        if Zvalues.shape[0] != 0:
            print( 'Level %f (min/max): '%level, Zvalues.min(), Zvalues.max() )

    print(X.shape)
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
    parser.add_argument( '-sp', '--sparseness', nargs=1, default = 1, type=int, help='points sparseness')
    main( parser.parse_args() )
