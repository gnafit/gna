#!/usr/bin/env python

from load import ROOT as R
import numpy as N
import itertools as I
from matplotlib import pyplot as plt
R.GNAObject

def plotl( title, step, X, Y, Z, **kwargs ):
    levels = kwargs.pop( 'levels', [] )
    masks  = kwargs.pop( 'masks', [] )
    assert len(levels)==len(masks)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( kwargs.pop( 'xlabel', 'axis x' ) )
    ax.set_ylabel( kwargs.pop('ylabel', 'axis y') )
    ax.set_title( kwargs.pop('title', 'Paraboloid\n' + title) )

    ax.contour( X, Y, Z, levels )

    markers = 'ov^s'
    X=X[::step, ::step]
    Y=Y[::step, ::step]
    for level, mask, marker in zip(levels, masks, I.cycle( markers ) ):
        ax.plot( X[mask], Y[mask], marker, label='L=%g'%level, alpha=0.06, markeredgecolor='none' )
        # there is a pdf bug for markers with transparency - visible gap between marker face and marker edge
        # therefore the edge color is disabled
    plt.subplots_adjust( top=0.87 )
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

def main( opts ):
    xsize = int(opts.xlinspace[2])
    ysize = int(opts.ylinspace[2])
    x = N.linspace( opts.xlinspace[0], opts.xlinspace[1], xsize, dtype='d' )
    y = N.linspace( opts.ylinspace[0], opts.ylinspace[1], ysize, dtype='d' )
    X, Y = N.meshgrid( x, y, indexing='ij' )
    Z = X**2 + Y**2
    Z1 = 2.0*(X-3.0)*(X-3.0) + 5.0*(Y-1.0)*(Y-1.0) + 3.0*(X-3.0)*(Y-1.0)
    Z2 = 0.05*X*X*X + 0.1*Y*Y*Y + Z1
    p = R.GridFilter(Z1.shape[0], Z1.shape[1], numpy_to_eigen( Z1 ), (opts.xlinspace[1] - opts.xlinspace[0]) / xsize, (opts.ylinspace[1] - opts.ylinspace[0]) / ysize, opts.deviation, opts.gradinfluence, opts.tolerance )
    masks = []
    title='deviation = ' + str(opts.deviation) + ', gradient influence = ' + str(opts.gradinfluence) + ',\ntolerance = ' + str(opts.tolerance) + ', points sparseness = ' + str(opts.sparseness)
    if opts.irregular:
        title = title + ', irregular deviation'
    for level in opts.levels:
	mat = R.Eigen.MatrixXd( xsize, ysize )
        if opts.irregular:
            p.GetCrossSectionExtendedIrregular( mat, level )
        else:
            p.GetCrossSectionExtendedAutoDev( mat, level )
        csc = eigen_to_numpy( mat, 'bool' ).copy()
        csc_sparse = csc[:: opts.sparseness, :: opts.sparseness]
        masks.append( csc_sparse )
        Zl=N.ma.array(Z1, mask=~csc)
        Zvalues = Zl.compressed()
        if Zvalues.shape[0] != 0:
            print( 'Level %f (min/max): '%level, Zvalues.min(), Zvalues.max() )

    plotl( title, opts.sparseness, X, Y, Z1, levels=opts.levels, masks=masks )

    plt.contour( X, Y, Z2, linestyles='--', levels=opts.levels)

    if opts.output:
        plt.savefig( opts.output )
        print( 'Save figure to:', opts.output )

    if opts.show:
        plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument( '-s', '--show', action='store_true', help='show figures' )
    parser.add_argument( '-irr', '--irregular', action='store_true' )
    parser.add_argument( '-l', '--levels', nargs='+', default=[1.0, 4.0, 9.0], type=float, help='levels to process')
    parser.add_argument( '-o', '--output', help='figure output' )
    parser.add_argument( '-sp', '--sparseness', type=int, default=1, help='points sparseness')
    parser.add_argument( '-tol', '--tolerance', type=float, default=0.05)
    parser.add_argument( '-dev', '--deviation', type=int, default=1, help='deviation multiplier')
    parser.add_argument( '-ginf', '--gradinfluence', type=float, default=1., help='multiplier for gradient')
    parser.add_argument( '--xlinspace', nargs=3, default=[0.0, 10.0, 501], type=float, help='parameters: BEGIN END NUM_OF_STEPS')
    parser.add_argument( '--ylinspace', nargs=3, default=[0.0, 10.0, 501], type=float, help='parameters: BEGIN END NUM_OF_STEPS')
    main( parser.parse_args() )
