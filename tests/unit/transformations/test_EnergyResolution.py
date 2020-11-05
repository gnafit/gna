#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
import gna.constructors as C
from gna import context
from mpl_tools import bindings
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from gna.env import env
from gna.labelfmt import formatter as L
from scipy.stats import norm
import os
from gna.unittest import *

from gna.converters import convert

def test_energyresolution_v01(tmp_path):
    def axes( title, ylabel='' ):
        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( L.u('evis') )
        ax.set_ylabel( ylabel )
        ax.set_title( title )
        return ax

    def singularities( values, edges ):
        indices = np.digitize( values, edges )-1
        phist = np.zeros( edges.size-1 )
        phist[indices] = 1.0
        return phist

    #
    # Define the parameters in the current namespace
    #
    ns = env.globalns('test_energyresolution_v01')
    weights = [ 'Eres_'+s for s in 'abc' ]
    wvals   = [0.016, 0.081, 0.026]
    percent = 0.01
    ns.defparameter(weights[0],  central=wvals[0], relsigma=30*percent )
    par = ns.defparameter(weights[1],  central=wvals[1], relsigma=30*percent )
    ns.defparameter(weights[2],  central=wvals[2], relsigma=30*percent )
    ns.printparameters()

    values = []
    def pop_value():
        values, par
        par.set(values.pop())

    def push_value(v):
        values, par
        values.append(par.value())
        par.set(v)

    #
    # Define bin edges
    #
    binwidth=0.05
    edges = np.arange( 0.0, 12.0001, binwidth )
    efine = np.arange( edges[0], edges[-1]+1.e-5, 0.005 )

    for eset in [
        [ [1.025], [3.025], [6.025], [9.025] ],
        [ [ 1.025, 5.025, 9.025 ] ],
        [ [ 6.025, 7.025,  8.025, 8.825 ] ],
        ]:
        for i, e in enumerate(eset):
            ax = axes( 'Energy resolution impact' )
            phist = singularities( e, edges )

            hist = C.Histogram( edges, phist )
            edges_o = R.HistEdges(hist)
            with ns:
                eres = C.EnergyResolution(weights, True)
            eres.matrix.Edges( hist )
            eres.smear.Ntrue( hist )

            path = os.path.join(str(tmp_path), 'eres_graph_%i.png'%i)
            savegraph(hist, path)
            allure_attach_file(path)

            smeared = eres.smear.Nrec.data()
            diff = phist.sum()-smeared.sum()
            print( 'Sum check for {} (diff): {}'.format( e, diff ) )
            assert diff<1.e-9

            lines = plot_hist( edges, smeared, label='default' )

            color = lines[0].get_color()
            ax.vlines( e, 0.0, smeared.max(), linestyle='--', color=color )

            if len(e)>1:
                color='green'
            for e in e:
                ax.plot( efine, binwidth*norm.pdf( efine, loc=e, scale=eres.relativeSigma(e)*e ), linestyle='--', color='green' )

            sprev = smeared.copy()
            push_value(0.162)
            assert eres.smear.tainted()
            smeared = eres.smear.Nrec.data()
            assert not np.all(smeared==sprev)
            plot_hist( edges, smeared, label='modified', color='red', alpha=0.5)
            pop_value()

            ax.legend()

            path = os.path.join(str(tmp_path), 'eres_test_{:02d}'.format(i))
            savefig(path, density=300)
            allure_attach_file(path)

            plt.close()

    smeared = eres.smear.Nrec.data()

    ax = axes( 'Relative energy uncertainty', ylabel=L.u('eres_sigma_rel') )
    ax.set_ylim(0, 13.0)
    ax.set_xlim(0.5, 12.0)
    x = np.arange( 0.5, 12.0, 0.01 )
    fcn = np.frompyfunc( eres.relativeSigma, 1, 1 )
    y = fcn( x )

    ax.plot( x, y*100. )
    path = os.path.join(str(tmp_path), 'eres_sigma')
    savefig(path, density=300)
    allure_attach_file(path)
    plt.close()

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( '' )
    ax.set_ylabel( '' )
    ax.set_title( 'Energy resolution convertsion matrix (class)' )

    mat = convert(eres.getDenseMatrix(), 'matrix')
    mat = np.ma.array( mat, mask= mat==0.0 )
    c = ax.matshow( mat, extent=[ edges[0], edges[-1], edges[-1], edges[0] ] )
    add_colorbar( c )

    path = os.path.join(str(tmp_path), 'eres_matc')
    savefig(path, density=300)
    allure_attach_file(path)
    plt.close()

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( '' )
    ax.set_ylabel( '' )
    ax.set_title( 'Energy resolution convertsion matrix (trans)' )

    eres.matrix.FakeMatrix.plot_matshow(colorbar=True, mask=0.0, extent=[edges[0], edges[-1], edges[-1], edges[0]])

    path = os.path.join(str(tmp_path), 'eres_mat')
    savefig(path, density=300)
    allure_attach_file(path)
    plt.close()

