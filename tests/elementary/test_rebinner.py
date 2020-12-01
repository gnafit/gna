#!/usr/bin/env python

from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C
import sys

def test_rebinner():
    edges   = N.array( [ 0.0, 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.0 ], dtype='d' )
    edges_m = N.array( [      0.1, 1.2,      3.4, 4.5,           7.8      ], dtype='d' )

    ntrue = C.Histogram(edges, N.ones( edges.size-1 ) )
    rebin = R.Rebin( edges_m.size, edges_m, 3 )
    rebin.rebin.histin( ntrue )

    olddata = ntrue.data()
    newdata = rebin.rebin.histout.data()

    mat = convert(rebin.getDenseMatrix(), 'matrix')

    if "pytest" not in sys.modules:
        print( mat )

    prj = mat.sum(axis=0)
    assert ((prj==1.0) + (prj==0.0)).all(), "Not matching results after rebin"

    #
    # Plot spectra
    #
    if "pytest" not in sys.modules:
        print( ((prj==1.0) + (prj==0.0)).all() and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( 'X axis' )
        ax.set_ylabel( 'Y axis' )
        ax.set_title( 'Rebinner' )

        ax.vlines( edges, 0.0, 4.0, linestyle='--', linewidth=0.5 )
        plot_hist( edges, olddata, label='before' )
        plot_hist( edges_m, newdata, label='after' )

        ax.legend( loc='upper left' )

        #
        # Plot matrix
        #
        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        # ax.grid()
        ax.set_xlabel( 'Source bins' )
        ax.set_ylabel( 'Target bins' )
        ax.set_title( 'Rebinning matrix' )

        c = ax.matshow( N.ma.array(mat, mask=mat==0.0), extent=[edges[0], edges[-1], edges_m[-1], edges_m[0]] )
        # add_colorbar( c )

        P.show()

if __name__ == "__main__":
    test_rebinner()
