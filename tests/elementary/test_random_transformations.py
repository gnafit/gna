#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test random matrices"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from converters import convert
import itertools as I
from mpl_tools.helpers import savefig, add_colorbar

mu    = N.array( [ 1.0, 10.0, 100.0 ] )
sigma = N.matrix( [ 1.0,  1.0, 10.0 ] )
cor   = N.array( [ [ 1.0, 0.5, 0.7 ],
                   [ 0.5, 1.0, 0.0 ],
                   [ 0.7, 0.0, 1.0 ] ], dtype='d' )
cov   = cor*(sigma.T*sigma).A

print( 'Covariance' )
print( cov )
print()

print( 'Correlation' )
print( cor )
print()

mu_p = convert(mu, 'points')
sigma_p = convert(sigma.A1, 'points')
cov_p = convert(cov, 'points')

chol = R.Cholesky()
chol.cholesky.mat( cov_p )

normal_cov = R.CovarianceToyMC( False )
normal     = R.NormalToyMC( False )
poisson    = R.PoissonToyMC( False )

normal_cov.add( mu_p, chol.cholesky.L )
normal.add( mu_p, sigma_p )
poisson.add( mu_p, chol.cholesky.L )

distrs = [ normal_cov, normal, poisson ]
labels = [ 'Normal (cov)', 'Normal (sigma)', 'Poisson' ]

def run(opts):
    for label, distr in zip( labels, distrs ):
        data = N.zeros( shape=opts.n, dtype=[ ('x', 'd'), ('y', 'd'), ('z', 'd') ] )

        out = R.Identity()
        out.identity.source( distr.toymc )

        for i in xrange( opts.n ):
            data[i] = out.identity.target.data()
            distr.nextSample()

        plot( label, data, opts )

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.set_xlabel( 'x, y, z' )
    ax.set_ylabel( 'x, y, z' )
    ax.set_title( 'Covmat' )

    c=ax.matshow( cov )
    add_colorbar( c )
    savefig( opts.output, suffix='_cov' )
    P.close()

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.set_xlabel( 'x, y, z' )
    ax.set_ylabel( 'x, y, z' )
    ax.set_title( 'Cormat' )

    c = ax.matshow( cor )
    add_colorbar( c )
    savefig( opts.output, suffix='_cor' )
    P.close()

def plot(label, distr, opts):
    print( label )
    it = zip(['x', 'y', 'z'], mu, sigma.A1)
    for k, m, s in it:
        if label=='Poisson':
            s = m**0.5
        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( k )
        ax.set_ylabel( 'entries' )
        ax.set_title( label )

        print( '  mean', k, distr[k].mean() )
        print( '  std', k, distr[k].std() )

        ax.hist( distr[k], bins=100, range=(m-5*s, m+5*s), histtype='stepfilled' )
        ax.axvline( m, linestyle='--', color='black' )
        ax.axvspan( m-s, m+s, facecolor='green', alpha=0.5, edgecolor='none' )

        savefig( opts.output, suffix=' %s %s'%(label, k) )
        P.close()

    for (k1, m1, s1), (k2, m2, s2) in I.combinations( it, 2 ):
        if label=='Poisson':
            s1 = m1**0.5
            s2 = m2**0.5

        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( k1 )
        ax.set_ylabel( k2 )
        ax.set_title( label )

        cc = N.corrcoef( distr[k1], distr[k2] )
        print( ' cor coef', k1, k2, cc[0,1])

        c,xe,ye,im=ax.hist2d( distr[k1], distr[k2], bins=100, cmin=0.5, range=[(m1-5*s1, m1+5*s1), (m2-5*s2, m2+5*s2)] )
        add_colorbar( im )
        ax.axvline( m1, linestyle='--', color='black'  )
        ax.axhline( m2, linestyle='--', color='black'  )

        savefig( opts.output, suffix=' %s %s %s'%(label, k1, k2) )
        P.close()

        if label=='Poisson':
            continue

        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( k1 )
        ax.set_ylabel( k2 )
        ax.set_title( label )

        cc = N.corrcoef( distr[k1], distr[k2] )
        print( ' cor coef', k1, k2, cc[0,1])

        c=ax.hexbin( distr[k1], distr[k2], gridsize=30, mincnt=1,
                     linewidths=0.0, edgecolor='none' )
        add_colorbar( c )
        ax.axvline( m1, linestyle='--', color='black'  )
        ax.axhline( m2, linestyle='--', color='black'  )

        savefig( opts.output, suffix=' %s %s %s hex'%(label, k1, k2) )
        P.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument( '-n', type=int, default=100, help='number of entries' )
    parser.add_argument( '-o', '--output', help='output file' )
    run(parser.parse_args())


