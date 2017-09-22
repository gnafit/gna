#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig

def axes( title, ylabel='' ):
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( L.u('evis') )
    ax.set_ylabel( ylabel )
    ax.set_title( title )
    return ax

def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

percent = 0.01
env.defparameter( 'Eres_a',  central=0.016, relsigma=30*percent )
env.defparameter( 'Eres_b',  central=0.081, relsigma=30*percent )
env.defparameter( 'Eres_c',  central=0.026, relsigma=30*percent )

binwidth=0.05
edges = N.arange( 0.0, 12.0, binwidth )

for eset in [
    [ 1.0, 3.0, 6.0, 9.0 ],
    [ ( 1.0, 5.0, 9.0 ) ],
    [ ( 6.0, 7.0,  8.0, 8.8  ) ],
    ]:
    ax = axes( 'Energy resolution impact' )
    for e in eset:
        phist = singularities( e, edges )

        hist = R.Histogram( phist.size, edges, phist )
        eres = R.EnergyResolution()
        eres.smear.Nvis( hist.hist )

        smeared = eres.smear.Nrec.data()
        print( 'Sum check for {} (diff): {}'.format( e, phist.sum()-smeared.sum() ) )

        alpha = 0.7
        bars = P.bar( edges[:-1], phist, binwidth, align='edge', alpha=alpha )
        P.bar( edges[:-1], smeared, binwidth, align='edge', alpha=alpha, color=bars[0].get_facecolor() )

ax = axes( 'Relative energy uncertainty', ylabel=L.u('eres_sigma_rel') )
x = N.arange( 0.5, 12.0, 0.01 )
fcn = N.frompyfunc( eres.relativeSigma, 1, 1 )
y = fcn( x )

ax.plot( x, y*100. )

P.show()
