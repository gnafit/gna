#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
R.SetMemoryPolicy(R.kMemoryStrict)
R.gDirectory.AddDirectory( False )
R.TH1.AddDirectory( False )
R.gSystem.Load('libGlobalNuAnalysis2.so')
R.GNAObject
import numpy as N
from matplotlib import pyplot as P
from mpl_tools import bindings
from mpl_tools.helpers import savefig, add_colorbar
from matplotlib.colors import LogNorm
from gna.labelfmt import formatter as L

options = None
def main( opts ):
    global options
    options = opts

    V  = N.diag( [ 100., 0.1, 0.3 ] )
    mu = N.array( [ 50.0, 0.5, 0.5 ], dtype='d' )
    sigma_r = N.array( [ 0.2, 0.2, 0.2 ], dtype='d' )

    #
    # Primary generator
    #
    V  = N.diag( [ 100., 0.1, 0.3 ] )
    mu = N.array( [ 50.0, 0.5, 0.5 ], dtype='d' )
    sigma_r = N.array( [ 0.1, 0.2, 0.2 ], dtype='d' )

    limit_lower = N.array( [   0.0, 0.0, 0.0] )
    limit_upper = N.array( [1000.0, 1.0, 1.0] )

    def generate( n ):
        raw_sample = generate_limited_distr( n,
                                             generator   = lambda n, mask=None: N.random.multivariate_normal( mu, V, size=n ),
                                             limit_lower = limit_lower,
                                             limit_upper = limit_upper )
        sigma=raw_sample*sigma_r
        det_sample = generate_limited_distr( n,
                                             generator   = lambda n, mask=None: N.random.normal( raw_sample, sigma ) if mask is None else N.random.normal( raw_sample[mask], sigma[mask] ),
                                             limit_lower = limit_lower,
                                             limit_upper = limit_upper
                                             )

        return raw_sample, det_sample

    #
    # Generate samples
    #
    sample_unf_true, sample_unf_det = generate( 1000000 )
    sample_data_true, sample_data_det = generate(  100000 )

    #
    # Init histograms
    #
    bins_true  = dict( bins=70, range=(20., 90.) )
    bins_rec   = dict( bins=80, range=(20., 100.) )
    # bins2 = dict( bins=(80, 100), range=[(20.0, 100.), (20.0, 100.)] )
    hist  = dict( histtype='bar', alpha=0.7 )

    mc_unf  = SimDistr( bins_true, sample_unf_true.T[0].copy(),  bins_rec, sample_unf_det.T[0].copy(),  label='unfold' )
    mc_data = SimDistr( bins_true, sample_data_true.T[0].copy(), bins_rec, sample_data_det.T[0].copy(), label='unfold' )

    mc_unf.init_unfolding()
    mc_data.set_unfolding( mc_unf )

    #
    # Plot
    #
    mc_unf.plot()
    mc_unf.plot2()

    mc_data.plot()
    mc_data.plot2()
    mc_data.plot_lcurve()

    P.show()

def generate_limited_distr( n, **kwargs ):
    generator=kwargs.get( 'generator' )
    distr = generator( n )
    fix_distr( distr, **kwargs )
    return distr

def fix_distr( distr, limit_lower, limit_upper, generator, **kwargs ):
    nmax = kwargs.pop( 'nmax', 100 )
    for i in xrange( nmax ):
        mask = ((distr<limit_lower)+(distr>limit_upper)).any( axis=1 )
        n_redo = mask.sum()
        # print( 'Iteration', i, '->', n_redo, 'to redo' )
        # print( distr )
        # print( n_redo, mask )
        # print()
        if not n_redo:
            break
        distr[mask] = generator( n_redo, mask )
    else:
        raise Exception( 'Reached maximal number of iterations' )

class SimDistr(object):
    ax, ax2 = None, None

    hist_unfolded, hist_folded = None, None
    lcurve = None
    def __init__(self, bins_true, data_true, bins_rec, data_rec, **kwargs):
        self.bins_true = bins_true
        self.data_true = data_true
        self.bins_rec  = bins_rec
        self.data_rec  = data_rec

        self.label = kwargs.pop( 'label', '' )

        self.hist_true = THist(  '%s_true'%self.label, 'true (%s)'%self.label,        self.bins_true,  self.data_true )
        self.hist_rec  = THist(  '%s_rec'%self.label,  'rec (%s)'%self.label,         self.bins_rec,   self.data_rec )
        self.hist2     = THist2( '%s2'%self.label,     'rec vs true (%s)'%self.label, self.bins_true,  self.data_true, self.bins_rec, self.data_rec )

    def init_unfolding(self):
        self.unfold = R.TUnfoldDensity(self.hist2, R.TUnfold.kHistMapOutputHoriz);

    def set_unfolding(self, o):
        if type(o) is R.TUnfold:
            self.unfold = o
        elif type(o) is SimDistr:
            self.unfold = o.unfold
        else:
            raise Exception( 'Unsupported object passed to set_unfolding method' )

        self.errcode = self.unfold.SetInput(self.hist_rec)
        print( 'Unfolding error code is', self.errcode )
        assert self.errcode<10000

        self.lcurve = R.ROOTHelpers.ScanLcurve( self.unfold, 100, 0.01, 3.0 )
        self.tau = R.ROOTHelpers.ScanTau( self.unfold, 100, 0.01, 3.0 )

        self.hist_unfolded = self.unfold.GetOutput( "unfolded" )
        self.hist_folded   = self.unfold.GetFoldedOutput( "folded" )

    def plot_lcurve(self):
        t, x, y = N.zeros( 1, dtype='d' ), N.zeros( 1, dtype='d' ), N.zeros( 1, dtype='d' )
        self.lcurve.logtaux.GetKnot(self.lcurve.ret, t, x)
        self.lcurve.logtauy.GetKnot(self.lcurve.ret, t, y)

        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( L('{^logL1_label}, {logL1}') )
        ax.set_ylabel( L('{^logL2_label}, {logL2}') )
        ax.set_title( 'L-curve' )

        self.lcurve.lcurve.plot()
        ax.plot( [x], [y], '*', label='choice' )

        ax.legend( loc='upper right' )
        savefig( options.output, suffix='_lcurve' )

        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( L('{logtau}') )
        ax.set_ylabel( '' )
        ax.set_title( '' )

        lx = self.lcurve.logtaux.plot( label=L('{logL1_label}') )
        ly = self.lcurve.logtauy.plot( label=L('{logL2_label}') )
        self.lcurve.logtauc.plot( label=r'curvature' )
        ax.plot( [t], [x], '*', color=lx[0].get_color() )
        ax.plot( [t], [y], '*', color=ly[0].get_color() )
        ax.axvline( t, linestyle='--' )

        ax.legend()
        savefig( options.output, suffix='_%s_lcurve_sub'%self.label )

    def plot_lcurve(self):
        t, x, y = N.zeros( 1, dtype='d' ), N.zeros( 1, dtype='d' ), N.zeros( 1, dtype='d' )
        self.lcurve.logtaux.GetKnot(self.lcurve.ret, t, x)
        self.lcurve.logtauy.GetKnot(self.lcurve.ret, t, y)

        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( L('{^logL1_label}, {logL1}') )
        ax.set_ylabel( L('{^logL2_label}, {logL2}') )
        ax.set_title( 'L-curve' )

        self.lcurve.lcurve.plot()
        ax.plot( [x], [y], '*', label='choice' )

        ax.legend( loc='upper right' )
        savefig( options.output, suffix='_lcurve' )

        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( L('{logtau}') )
        ax.set_ylabel( '' )
        ax.set_title( '' )

        lx = self.lcurve.logtaux.plot( label=L('{logL1_label}') )
        ly = self.lcurve.logtauy.plot( label=L('{logL2_label}') )
        self.lcurve.logtauc.plot( label=r'curvature' )
        ax.plot( [t], [x], '*', color=lx[0].get_color() )
        ax.plot( [t], [y], '*', color=ly[0].get_color() )
        ax.axvline( t, linestyle='--' )

        ax.legend()
        savefig( options.output, suffix='_%s_lcurve_sub'%self.label )

    def plot(self, ax=None):
        if ax:
            P.sca( ax )
        else:
            fig = P.figure()
            ax = P.subplot( 111 )
            ax.minorticks_on()
            ax.grid()
            ax.set_xlabel( 'variable' )
            ax.set_ylabel( 'entries' )
            ax.set_title( 'Unfolding MC' )
        self.ax = ax

        self.hist_true.plot( label='True' )
        self.hist_rec.plot( label='Detected' )

        if self.hist_unfolded:
            self.hist_unfolded.plot( label='Unfolded' )

        # if self.hist_folded:
            # self.hist_folded.plot( label='Folded back' )

        ax.legend( loc='upper right' )
        savefig( options.output, suffix='_%s_hist'%self.label )

    def plot2(self, ax2=None):
        if ax2:
            P.sca( ax )
        else:
            fig = P.figure()
            ax2 = P.subplot( 111 )
            ax2.minorticks_on()
            ax2.set_xlabel( 'true variable' )
            ax2.set_ylabel( 'reconstructed variable' )
            ax2.set_title( 'Unfolding hist' )
        self.ax2 = ax2
        P.sca( self.ax2 )

        kwargs = dict()
        # kwargs['norm']=LogNorm( vmin=1 )
        self.hist2.pcolorfast( colorbar=True, mask=0.0, **kwargs )
        savefig( options.output, suffix='_%s_hist2'%self.label )

        return ax2

def THist( name, label, bins, data ):
    hist = R.TH1D( name, label, bins['bins'], bins['range'][0], bins['range'][1] )
    hist.FillN( data.size, data, N.ones( data.shape, data.dtype ) )
    return hist

def THist2( name, label, binsx, datax, binsy, datay ):
    hist = R.TH2D( name, label, binsx['bins'], binsx['range'][0], binsx['range'][1],
                                binsy['bins'], binsy['range'][0], binsy['range'][1] )
    hist.FillN( datax.size, datax, datay, N.ones( datax.shape, datax.dtype ) )
    return hist

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--erange', default=[ 10.0, 200.0, 19.0 ], nargs=3, type=float, help='energy bins', metavar=('emin', 'emax', 'nbins'))
    parser.add_argument('--nunfold', type=int, default=100, help='number of ToyMC unfolding samples')
    parser.add_argument('--ndata',   type=int, default=10,  help='number of ToyMC data samples')
    parser.add_argument('-o' ,'--output', help='output filename')

    main( parser.parse_args() )
