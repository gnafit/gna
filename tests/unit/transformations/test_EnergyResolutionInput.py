#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
import gna.constructors as C
from mpl_tools import bindings
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from gna.labelfmt import formatter as L
from scipy.stats import norm
import os
from gna.unittest import allure_attach_file, savegraph
import pytest
import time
from typing import Any

def check_smearing_projection(mat: np.ndarray, *, check_assert: bool=True) -> None:
    threshold = 1.e-8
    ones = mat.sum(axis=0)
    zeros = np.fabs(ones-1.0)

    istart, iend = None, None
    for istart, val in enumerate(zeros):
        if val<threshold:
            break

    for iend, val in enumerate(reversed(zeros)):
        if val<threshold:
            iend = -iend
            break

    if iend==0:
        iend = None

    zerossub = zeros[istart:iend]

    if check_assert:
        assert (zerossub<threshold).all()

    return ones

# @pytest.mark.parametrize('classmark', [ 'BinCenter', 'BinCenter2', 'Erf', 'Erf2'][-2:-1])
# @pytest.mark.parametrize('sigmaopt', [ 'abc', 'narrow', 'raising'][-1:])
# @pytest.mark.parametrize('input_type', [ 'none', 'hist', 'points'][:1])
# @pytest.mark.parametrize('input_binning', ['equal', 'variable'][:1])
# @pytest.mark.parametrize('output_binning', ['same', 'stride', 'offset', 'strideoffset'][:1])
@pytest.mark.parametrize('classmark', [ 'BinCenter', 'BinCenter2', 'Erf', 'Erf2'])
@pytest.mark.parametrize('sigmaopt', [ 'abc', 'narrow', 'raising'])
@pytest.mark.parametrize('input_type', [ 'none', 'hist', 'points'])
@pytest.mark.parametrize('input_binning', ['equal', 'variable'])
@pytest.mark.parametrize('output_binning', ['same', 'stride', 'offset', 'strideoffset'])
def test_energyresolutioninput_v01(classmark, sigmaopt, input_type, input_binning, output_binning, tmp_path):
    if classmark in ['BinCenter', 'Erf']:
        if output_binning!='same':
            return
        if input_type!='none':
            return
    elif classmark in ['Erf2', 'BinCenter2']:
        if input_type=='none':
            return
    EresClasses = {
            'BinCenter': C.EnergyResolutionInput,
            'BinCenter2': C.EnergyResolutionInputs,
            'Erf': C.EnergyResolutionErfInput,
            'Erf2': C.EnergyResolutionErfInputs,
            }
    Class = EresClasses[classmark]
    def axes(title, ylabel=''):
        extra = f': {classmark}, {input_type}, {sigmaopt}, {input_binning}, {output_binning}'

        plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( L.u('evis') )
        ax.set_ylabel( ylabel )
        ax.set_title( title+extra )
        return ax

    def singularities( values, edges ):
        indices = np.digitize( values, edges )-1
        phist = np.zeros( edges.size-1 )
        phist[indices] = 1.0
        return phist

    suffix = f'{classmark}_{input_binning}_{output_binning}_{input_type}_{sigmaopt}'

    #
    # Define bin edges
    #
    if input_binning=='equal':
        binwidth_in=0.05
        edges_in = np.arange( 0.0, 12.0001, binwidth_in )
    else:
        edges_in = np.geomspace(1.0, 12.0, 200)
        binwidth_in = edges_in[1:] - edges_in[:-1]

    efine = np.arange( edges_in[0], edges_in[-1]+1.e-5, 0.005 )
    centers_in = 0.5*(edges_in[1:]+edges_in[:-1])

    if output_binning=='same':
        edges_out = edges_in
        # centers_out = centers_in
    else:
        if output_binning=='stride':
            edges_out = edges_in[::2]
        elif output_binning=='offset':
            if isinstance(binwidth_in, float):
                edges_out=edges_in+binwidth_in/3.0
            else:
                edges_out = edges_in.copy()
                edges_out[:-1]+=binwidth_in/3.0
                edges_out[-1]+=binwidth_in[-1]/3.0
        else:
            if isinstance(binwidth_in, float):
                edges_out = edges_in[::2]+binwidth_in*0.5
            else:
                edges_out = edges_in[::2]+binwidth_in[::2]*0.5
        # centers_out = 0.5*(edges_out[1:]+edges_out[:-1])

    binwidth_out = edges_out[1:] - edges_out[:-1]

    # print('Edges out mode:', output_binning)
    # print('Edges in:', edges_in)
    # print('Edges out:', edges_out)

    wvals = [0.016, 0.081, 0.026]
    if sigmaopt=='abc':
        def RelSigma(e):
            a, b, c = wvals
            return (a**2+ (b**2)/e + (c/e)**2)**0.5
    elif sigmaopt=='narrow':
        def RelSigma(e):
            return 0.1*binwidth_in/e
    elif sigmaopt=='raising':
        def RelSigma(e):
            return (np.fabs(e-1.0)*0.019 + 0.01)/e
    else:
        assert False

    relsigma = RelSigma(centers_in)
    abssigma = relsigma*centers_in

    energy_sets = [
        [ [1.025], [3.025], [6.025], [9.025] ],
        [ [ 1.025, 5.025, 9.025 ] ],
        [ [ 6.025, 7.025,  8.025, 8.825 ] ],
        ]
    energy_sets = [[[1.025, 5.025, 9.025]]]

    multiple_sets = len(energy_sets)>1

    hist_out = C.Histogram(edges_out)
    points_in = C.Points(edges_in)

    collection=[]
    eres: Any = None
    for iset, energy_set in enumerate(energy_sets):
        for i, energies in enumerate(energy_set):
            ax = axes( 'Energy resolution (input) impact' )
            phist_in = singularities( energies, edges_in )
            relsigma_i = relsigma.copy()
            relsigma_p = C.Points(relsigma_i)
            collection.append(relsigma_p)

            hist_in = C.Histogram(edges_in, phist_in)
            collection.append(hist_in)
            # edges_o = R.HistEdges(hist_in)
            if classmark=='BinCenter':
                eres = Class(R.GNA.DataPropagation.Propagate)
            else:
                args = ()
                if input_type=='points':
                    args = (R.GNA.DataMutability.Dynamic, )
                elif input_type=='hist':
                    args = (R.GNA.DataMutability.Static, )
                else:
                    assert input_type=='none'
                eres = Class(*args)
            collection.append(eres)

            if input_type=='hist':
                hist_out >> eres.matrix.HistEdgesOut
                hist_in >> eres.matrix.EdgesIn
            elif input_type=='points':
                hist_out >> eres.matrix.HistEdgesOut
                points_in >> eres.matrix.EdgesIn
            else:
                hist_in >> eres.matrix.Edges
            relsigma_p >> eres.matrix.RelSigma
            hist_in >> eres.smear.Ntrue

            # eres.printtransformations()

            ntimes = 0
            if ntimes:
                #
                # Measure performance
                #
                mtrans = eres.matrix
                etime = time.perf_counter_ns()
                for i in range(ntimes):
                    mtrans.taint()
                    mtrans.touch()
                etime = time.perf_counter_ns()-etime
                etime/=1e6*ntimes
                print()
                print(f'{classmark} {sigmaopt} {iset}.{i}: {etime} ms')

            # thresholds = {
            #         'BinCenter':  1.e-9,
            #         'Erf':        1.e-8,
            #         'Erf2hist':   1.e-8,
            #         'Erf2points': 1.e-8,
            #         }
            # threshold = thresholds[classmark]
            smeared = eres.smear.Nrec.data()
            # diff = np.fabs(phist_in.sum()-smeared.sum())
            # print( f'Sum check for {energies} (diff): {diff} < {threshold}' )
            # if not (classmark=='BinCenter' and sigmaopt=='narrow'):
            #     # assert diff<threshold
            #     pass

            lines = plot_hist( edges_in, hist_in.data(), label='in' )

            lines = plot_hist( edges_out, smeared, label='default' )
            maxv = smeared.max()
            if not np.isnan(maxv):
                # ax.set_ylim(0.0, maxv*1.1)

                color = lines[0].get_color()
                ax.vlines( energies, 0.0, maxv, linestyle='--', color=color )
            ax.set_ylim(0.0, 1.0)

            if len(energies)>1:
                color='green'

            if input_binning=='equal':
                for energy in energies:
                    ax.plot( efine, binwidth_out[0]*norm.pdf( efine, loc=energy, scale=RelSigma(energy)*energy ), linestyle='--', color='green' )

            sprev = smeared.copy()

            icut = relsigma_i.size//2
            relsigma_i[icut:]*=2
            relsigma_p.set(relsigma_i, relsigma_i.size)
            smeared = eres.smear.Nrec.data()
            shouldchange = phist_in[icut:].any()
            if not (classmark.startswith('BinCenter') and sigmaopt=='narrow'):
                assert not np.all(smeared==sprev)==shouldchange
            plot_hist( edges_out, smeared, label='modified', color='red', alpha=0.5)

            ax.legend()

            path = os.path.join(str(tmp_path), f'eres_test_{suffix}_{i:02d}.png')
            savefig(path, dpi=300)
            allure_attach_file(path)
            if multiple_sets:
                plt.close('all')

            relsigma_p.set(relsigma, relsigma.size)

    smeared = eres.smear.Nrec.data()

    path = os.path.join(str(tmp_path), f'eres_graph_{suffix}.png')
    savegraph(eres.smear.Nrec, path)
    allure_attach_file(path)

    ax = axes( f'Relative energy uncertainty', ylabel=L.u('eres_sigma_rel') )
    ax.set_xlim(0.5, 12.0)
    # ax.set_ylim(0, 13.0)

    ax.plot( centers_in, relsigma*100. )
    path = os.path.join(str(tmp_path), f'eres_sigma_rel_{suffix}.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    # plt.close('all')

    ax = axes( 'Absolute energy uncertainty', ylabel=L.u('eres_sigma_abs') )
    # ax.set_xlim(0.5, 12.0)
    # ax.set_ylim(0, 13.0)

    ax.plot( centers_in, abssigma )
    path = os.path.join(str(tmp_path), f'eres_sigma_abs_{suffix}.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel(f'E true, {edges_in.size-1} bins')
    ax.set_ylabel(f'E smeared, {edges_out.size-1} bins')
    ax.set_title(f'Matrix: {classmark}, {sigmaopt}, {output_binning}, {input_binning}')

    # mat = eres.getDenseMatrix()
    # mat = convert(mat, 'matrix')
    mat = eres.matrix.FakeMatrix.data()
    ones = check_smearing_projection(mat, check_assert=(classmark!='BinCenter2'))
    mat = np.ma.array( mat, mask= mat==0.0 )

    if output_binning=='same':
        c = ax.matshow(mat, extent=[ edges_in[0], edges_in[-1], edges_in[-1], edges_in[0] ])

        ax.tick_params(bottom=True, labelbottom=True)
    else:
        c = ax.pcolorfast(edges_in, edges_out, mat)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.tick_params(top=True, labeltop=True)
    add_colorbar( c )

    path = os.path.join(str(tmp_path), f'eres_mat_{suffix}.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
    # plt.close('all')

    plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel(f'E true, {edges_in.size-1} bins')
    ax.set_ylabel(f'Col sum')
    ax.set_title(f'Norm: {classmark}, {sigmaopt}, {output_binning}, {input_binning}')

    plot_hist(edges_in, ones)
    ax.set_ylim(0.99, 1.01)

    path = os.path.join(str(tmp_path), f'eres_colsum_{suffix}.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    if output_binning=='same':
        plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( 'E, MeV' )
        ax.set_ylabel( 'fraction' )
        ax.set_title( f'E res. conversion matrix diagonal: {classmark}, {sigmaopt}' )

        binwidth_to_sigma_in = binwidth_in/abssigma
        diag0 = mat.diagonal()
        diag1 = mat.diagonal(offset=1)
        diag2 = mat.diagonal(offset=2)
        diag3 = mat.diagonal(offset=3)

        # eres.matrix.FakeMatrix.plot_matshow(colorbar=True, mask=0.0, extent=[edges_in[0], edges_in[-1], edges_in[-1], edges_in[0]])
        plot_hist(edges_in, diag0, label='of events (diagonal)')
        plot_hist(edges_in[1:], diag1, label='of events (diagonal 1)')
        plot_hist(edges_in[2:], diag2, label='of events (diagonal 2)')
        plot_hist(edges_in[3:], diag3, label='of events (diagonal 3)')
        ax.plot(centers_in, binwidth_to_sigma_in, label=r'$w/\sigma$')

        ax.legend()

        path = os.path.join(str(tmp_path), f'eres_matdiag_{suffix}.png')
        savefig(path, dpi=300)
        allure_attach_file(path)

        plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( r'$w/\sigma$' )
        ax.set_ylabel( 'fraction' )
        # ax.set_title( f'Energy resolution conversion matrix diagonal')#: {classmark}, {sigmaopt}' )
        ax.set_title( f'Energy resolution conversion matrix diagonal: {classmark}, {sigmaopt}' )

        # eres.matrix.FakeMatrix.plot_matshow(colorbar=True, mask=0.0, extent=[edges_in[0], edges_in[-1], edges_in[-1], edges_in[0]])
        ax.plot(binwidth_to_sigma_in,     diag0, label='same bin')
        ax.plot(binwidth_to_sigma_in[1:], 2.0*diag1, label='1st neighbour bins')
        ax.plot(binwidth_to_sigma_in[2:], 2.0*diag2, label='2nd neighbour bins')
        ax.plot(binwidth_to_sigma_in[3:], 2.0*diag3, label='3d neighbour bins')
        ax.plot(binwidth_to_sigma_in[3:], 2.0*(diag1[2:]+diag2[1:]+diag3), label='1st-3d neighbour bins combined')
        ax.set_xlim(left=0)
        ax.set_ylim(0.0, 1.0)

        ax.legend(title='fraction of events in', ncol=2)

        path = os.path.join(str(tmp_path), f'eres_matdiag1_{suffix}.png')
        # path = os.path.join(str(tmp_path), f'eres_matdiag1_{suffix}.pdf')
        savefig(path, dpi=300)
        allure_attach_file(path)

    # plt.show()
    plt.close('all')

