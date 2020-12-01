#!/usr/bin/env python

import load
from matplotlib import pyplot as plt
import numpy as np
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.interpolate import interp1d
from argparse import ArgumentParser
import gna.constructors as C

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-o', '--output', help='output file')
parser.add_argument('-s', '--show', action='store_true', help='output file')
opts = parser.parse_args()

def cmp(a, b):
    good = np.allclose(a, b, rtol=0, atol=1e-15)
    print( good and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
    if not good:
        diff = a-b
        print( 'diff' )
        print( diff )
        print()
        print(diff)

def rescale_to_matrix_a(edges_from, edges_to, **kwargs):
    """Use only modified edges"""
    roundto = kwargs.pop( 'roundto', None )
    if not roundto is None:
        edges_from = np.round( edges_from, roundto )
        edges_to   = np.round( edges_to,   roundto )
    skipv = kwargs.pop( 'skip_values', [] )
    assert not kwargs

    idx = np.searchsorted( edges_from, edges_to, side='right' )-1
    widths = edges_to[1:]-edges_to[:-1]

    mat = np.zeros( shape=(edges_from.shape[0]-1, edges_from.shape[0]-1) )
    i1s = np.maximum( 0, idx[:-1] )
    i2s = np.minimum( idx[1:], edges_from.size-2 )
    for j, (i1, i2) in enumerate(zip(i1s, i2s)):
        if i2<0 or i1>=edges_from.size or edges_to[j]<-1.e100: continue
        for i in range( i1, i2+1 ):
            if not (0<=i<mat.shape[0] and 0<=j<mat.shape[1]):
                continue
            l1 = max( edges_to[j],   edges_from[i] )
            l2 = min( edges_to[j+1], edges_from[i+1] )
            w  = (l2-l1)/widths[j]

            mat[i,j] = w

    return mat

def rescale_to_matrix_b(edges_from, centers_to, **kwargs):
    """Use only modified centers"""
    roundto = kwargs.pop( 'roundto', None )
    if not roundto is None:
        edges_from = np.round(edges_from, roundto)
        centers_to = np.round(centers_to,   roundto)

    widths = edges_from[1:]-edges_from[:-1]
    hwidths = 0.5*widths

    skipv = kwargs.pop( 'skip_values', [] )
    assert not kwargs

    edges_to_left = centers_to-hwidths
    edges_to_right = centers_to+hwidths
    idx_left = np.searchsorted(edges_from, edges_to_left, side='right')-1
    idx_right = np.searchsorted(edges_from, edges_to_right, side='right')-1

    mat = np.zeros( shape=(edges_from.shape[0]-1, edges_from.shape[0]-1) )
    i1s = np.maximum( 0, idx_left )
    i2s = np.minimum( idx_right, edges_from.size-2 )
    for j, (i1, i2) in enumerate(zip(i1s, i2s)):
        if i2<0 or i1>=edges_from.size or centers_to[j]<-1.e100: continue
        for i in range( i1, i2+1 ):
            if not (0<=i<mat.shape[0] and 0<=j<mat.shape[1]):
                continue
            l1 = max( edges_to_left[j],  edges_from[i] )
            l2 = min( edges_to_right[j], edges_from[i+1] )
            w  = (l2-l1)/widths[j]

            mat[i,j] = w

    return mat

def rescale_to_matrix_c(edges_from, edges_to, centers_to, **kwargs):
    """Use both modified edges and centers to do asymmetric scaling"""
    roundto = kwargs.pop( 'roundto', None )
    if not roundto is None:
        edges_from = np.round( edges_from, roundto )
        edges_to   = np.round( edges_to,   roundto )
    skipv = kwargs.pop( 'skip_values', [] )
    assert not kwargs

    idx = np.searchsorted( edges_from, edges_to, side='right' )-1
    widths_left  = 2.0*(centers_to-edges_to[:-1])
    widths_right = 2.0*(edges_to[1:]-centers_to)

    mat = np.zeros( shape=(edges_from.shape[0]-1, edges_from.shape[0]-1) )
    i1s = np.maximum( 0, idx[:-1] )
    i2s = np.minimum( idx[1:], edges_from.size-2 )
    for j, (i1, i2) in enumerate(zip(i1s, i2s)):
        if i2<0 or i1>=edges_from.size or edges_to[j]<-1.e100: continue
        for i in range( i1, i2+1 ):
            if not (0<=i<mat.shape[0] and 0<=j<mat.shape[1]):
                continue

            l1 = max( edges_to[j],   edges_from[i] )
            l2 = min( edges_to[j+1], edges_from[i+1] )

            center = centers_to[j]
            if l2<=center:
                w = (l2-l1)/widths_left[j]
            elif l1>=center:
                w = (l2-l1)/widths_right[j]
            else:
                w = (center-l1)/widths_left[j]
                w+= (l2-center)/widths_right[j]

            mat[i,j] = w

    return mat


edges  = np.array( [ -1.0,   1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0  ] )
factor = np.array( [ -2e100, 0.8, 0.9, 0.95, 1.05, 1.10, 1.15, 1.20 ] )
centers = 0.5*(edges[1:] + edges[:-1])

conversion_fcn_direct  = interp1d(edges[1:], edges[1:]*factor[1:], kind='cubic', bounds_error=False, fill_value=factor[0])
conversion_fcn_inverse = interp1d(edges[1:]*factor[1:], edges[1:], kind='cubic', bounds_error=False, fill_value=factor[0])
centers_m = conversion_fcn_direct(centers)
edges_m = conversion_fcn_direct(edges)

edges_back = conversion_fcn_inverse(edges)

print('Edges before:', edges)
print('Edges after:', edges_m)
print('Edges back:', edges_back)

matp_a = rescale_to_matrix_a( edges, edges_m, roundto=3 )
matp_b = rescale_to_matrix_b( edges, centers_m, roundto=3 )
matp_c = rescale_to_matrix_c( edges, edges_m, centers_m, roundto=3 )

pedges_m = C.Points(edges_m)
pcenters = C.Points(centers)
ntrue = C.Histogram(edges, np.ones(edges.size-1))

nlA = C.HistNonlinearity(True)
nlA.set(ntrue.hist, pedges_m)
nlA.add_input()

# nlB = C.HistNonlinearityB(True)
# nlB.set(ntrue.hist, pcenters)
# nlB.add_input()

mat_a = nlA.matrix.FakeMatrix.data()
# mat_b = nlB.matrix.FakeMatrix.data()

print( 'Mat A and its sum' )
print( mat_a )
print( mat_a.sum( axis=0 ) )

print( 'Mat B and its sum' )
print( matp_b )
print( matp_b.sum( axis=0 ) )

print( 'Mat C and its sum' )
print( matp_c )
print( matp_c.sum( axis=0 ) )

cmp(mat_a, matp_a)

ntrue.hist.hist.data()
tflag1a = ntrue.hist.tainted()
tflag2a = nlA.matrix.tainted()
ntrue.hist.taint()
tflag1b = ntrue.hist.tainted()
tflag2b = nlA.matrix.tainted()
print('taints', tflag1a, tflag2a, tflag1b, tflag2b)
assert not tflag2b

def plot_edges_a(edges, centers, edges_m, centers_m):
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title='Edges A')
    ax.set_xlim(left=-2.0, right=10.0)
    ax.set_ylim(0.0, 3.0)

    ax.vlines(edges,   0.0, 5.0, linestyle='solid',  color='black', alpha=0.4, linewidth=0.2)
    ax.vlines(edges,   0.0, 1.0, linestyle='solid',  color='blue', label='Original')
    ax.vlines(centers, 0.0, 1.0, linestyle='dashed', color='blue')

    ax.vlines(edges_m,   2.0, 3.0, linestyle='solid',  color='red', label='Modified')
    ax.vlines(centers_m, 2.0, 3.0, linestyle='dashed', color='red')

    for a, b in zip(edges, edges_m):
        ax.plot( (a, b), (1.0, 2.0), color='gray' )

    for i, c in enumerate(centers):
        ax.text(c, 1.33, str(i), va='center', ha='center')

    for i, c in enumerate(centers_m):
        ax.text(c, 1.66, str(i), va='center', ha='center')

    ax.legend(loc='upper left')

    savefig(opts.output, suffix='edges_a')

def plot_edges_b(edges, centers, edges_m, centers_m):
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title='Edges B')
    ax.set_xlim(left=-2.0, right=10.0)
    ax.set_ylim(0.0, 3.0)

    ax.vlines(edges,   0.0, 5.0, linestyle='solid',  color='black', alpha=0.4, linewidth=0.2)
    ax.vlines(edges,   0.0, 1.0, linestyle='solid',  color='blue', label='Original')
    ax.vlines(centers, 0.0, 1.0, linestyle='dashed', color='blue')

    hwidths = 0.5*(edges[1:]-edges[:-1])
    edges_m1 = centers_m-hwidths
    edges_m2 = centers_m+hwidths
    ax.vlines(edges_m1,  2.0, 2.5, linestyle='solid',  color='red', label='Left')
    ax.vlines(edges_m2,  2.5, 3.0, linestyle='solid',  color='magenta', label='Right')
    ax.vlines(centers_m, 2.0, 3.0, linestyle='dashed', color='red')

    for a, b in zip(edges, edges_m):
        ax.plot( (a, b), (1.0, 2.0), color='gray' )

    for i, c in enumerate(centers):
        ax.text(c, 1.33, str(i), va='center', ha='center')

    for i, c in enumerate(centers_m):
        ax.text(c, 1.66, str(i), va='center', ha='center')

    ax.legend(loc='upper left')

    savefig(opts.output, suffix='edges_b')

def plot_matrix(mat, title, suffix):
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.set_xlabel( 'Source bins' )
    ax.set_ylabel( 'Target bins' )
    ax.set_title( 'Bin edges conversion matrix: {}'.format(title) )

    mmat = np.ma.array(mat, mask=mat==0.0)
    c = ax.matshow(mmat, extent=[edges[0], edges[-1], edges[-1], edges[0]])
    add_colorbar(c)

    savefig(opts.output, suffix='matrix_'+suffix)

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='Edges', ylabel='', title='Check sum: {}'.format(title))
    ax.minorticks_on()

    rsum = mat.sum(axis=0)
    ax.plot(centers, rsum, 'o')
    ax.axhline(1.0)

    savefig(opts.output, suffix='norm_'+suffix)

def plot_edges():
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title='')
    ax.minorticks_on()
    ax.grid()

    ax.plot(edges[1:],      edges_m[1:], 'o-', label='direct')
    ax.plot(edges_back[1:], edges[1:],   'o-', label='inverse')
    ax.plot(edges[1:],      edges_back[1:], 'o-', label='inverse')

    ax.legend()

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title='')
    ax.minorticks_on()
    # ax.grid()

    ax.vlines(edges[1:],      0.0, 1.0, linestyle='solid', label='Original')
    ax.vlines(edges_m[1:],    1.0, 2.0, linestyle='solid', label='Modified')
    ax.vlines(edges_back[1:], 2.0, 3.0, linestyle='solid', label='Modified')

    ax.legend()



plot_edges_a(edges, centers, edges_m, centers_m)
plot_edges_b(edges, centers, edges_m, centers_m)

plot_matrix(mat_a, 'A, edges', 'a')
plot_matrix(matp_b, 'B, centers', 'b')
plot_matrix(matp_c, 'C, edges+centers', 'c')

plot_edges()

if opts.show:
    plt.show()
