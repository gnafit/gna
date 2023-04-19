#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
import gna.constructors as C
from mpl_tools.helpers import savefig, add_colorbar
from scipy.linalg import block_diag
from typing import Union, Optional, Tuple
import itertools as it
from scipy.interpolate import interp1d
from gna.unittest import allure_attach_file
import os

def test_energyresolutionerf_consistency(tmp_path) -> None:
    eleft, eright = 0.0, 12.0
    nbins_a  = 300
    edges_a = np.linspace(eleft, eright, nbins_a+1)

    factors_ab= 11, 1
    factors_bc = 2

    assert isinstance(factors_bc, int)
    if isinstance(factors_ab, int):
        factors_ac = factors_ab*factors_bc
        weights_ac = None
    else:
        assert factors_bc==len(factors_ab)
        factors_ac = sum(factors_ab)
        weights_ac = tuple(f/factors_ac for f in factors_ab)

    def mergeedges(edges: np.ndarray, ntogroup: Union[int, Tuple[int,...]]) -> np.ndarray:
        if isinstance(ntogroup, int):
            return edges[::ntogroup]
        if len(ntogroup)==1:
            return edges[::ntogroup[0]]

        nblock = sum(ntogroup)
        ret = np.sort(np.concatenate(tuple(edges[offset::nblock] for offset in it.accumulate((0,)+ntogroup[:-1])))).astype('d')

        return ret

    edges_b = mergeedges(edges_a, factors_ab)
    nbins_b = edges_b.size-1

    edges_c = mergeedges(edges_b, factors_bc)
    nbins_c = edges_c.size-1

    edges_all = {'a' :edges_a, 'b': edges_b, 'c': edges_c}

    abssigma_edges_c = np.linspace(0.1, 1.0, edges_c.size)
    abssigma_fcn = interp1d(edges_c, abssigma_edges_c, kind='previous')
    relsigma_fcn = lambda e: abssigma_fcn(e)/e

    centers_all = tuple((e[1:]+e[:-1])*0.5 for e in edges_all.values())
    # centers_a, centers_b, centers_c = centers_all
    relsigma_all = tuple(relsigma_fcn(c) for c in centers_all)

    print(f'{factors_ab=}')
    print(f'{factors_bc=}')
    print(f'{factors_ac=}')
    print(f'{weights_ac=}')

    for k, edges in edges_all.items():
        print(f'Edges {k}:', edges.shape, edges)

    # for k, relsigma in zip(edges_all, relsigma_all):
    #     print(f'σ/E {k}:', relsigma)

    RelSigma_a, RelSigma_b, RelSigma_c = (C.Points(s) for s in relsigma_all)
    Hist_a, Hist_b, Hist_c = (C.Histogram(e, np.ones(e.size-1, dtype='d')) for e in edges_all.values())

    Eres_a0   = C.EnergyResolutionInput(R.GNA.DataPropagation.Propagate)
    Eres_aa0  = C.EnergyResolutionInputs()
    Eres_a    = C.EnergyResolutionErfInput()
    Eres_aa   = C.EnergyResolutionErfInputs()
    Eres_ab   = C.EnergyResolutionErfInputs()
    Eres_bb   = C.EnergyResolutionErfInputs()
    Eres_bc   = C.EnergyResolutionErfInputs()
    Eres_cb   = C.EnergyResolutionErfInputs()
    Eres_cc   = C.EnergyResolutionErfInputs()

    eres_all = {
            'a0': Eres_a0,
            'aa0': Eres_aa0,
            'a':  Eres_a,
            'aa': Eres_aa, 'ab': Eres_ab, # 'ac': Eres_ac,
            'bb': Eres_bb, 'bc': Eres_bc,
            'cc': Eres_cc,
            'cb': Eres_cb
            }

    verbose = False
    def bind(eres_ab, hist_a, hist_b, relsigma_a):
        if hist_b is None:
            hist_a     >> eres_ab.matrix.Edges
            relsigma_a >> eres_ab.matrix.RelSigma
        else:
            hist_a     >> eres_ab.matrix.EdgesIn
            hist_b     >> eres_ab.matrix.HistEdgesOut
            relsigma_a >> eres_ab.matrix.RelSigma

        hist_a >> eres_ab.transformations.smear.Ntrue

        if verbose:
            eres_ab.printtransformations()

    bind(Eres_a0, Hist_a, None,   RelSigma_a)
    bind(Eres_a,  Hist_a, None,   RelSigma_a)
    bind(Eres_aa0, Hist_a, Hist_a, RelSigma_a)
    bind(Eres_aa, Hist_a, Hist_a, RelSigma_a)
    bind(Eres_ab, Hist_a, Hist_b, RelSigma_a)
    bind(Eres_bb, Hist_b, Hist_b, RelSigma_b)
    bind(Eres_bc, Hist_b, Hist_c, RelSigma_b)
    bind(Eres_cb, Hist_c, Hist_b, RelSigma_c)
    bind(Eres_cc, Hist_c, Hist_c, RelSigma_c)

    plot_counter = [0]
    def plotmat(mat: np.ndarray, name: str, edges_a: Optional[np.ndarray]=None, edges_b: Optional[np.ndarray]=None, plot_counter=plot_counter):
        plt.figure()
        ax = plt.subplot(111)
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel(f'E true, {mat.shape[1]} bins')
        ax.set_ylabel(f'E smeared, {mat.shape[0]} bins')
        ax.set_title(f'Resolution matrix {name}')
        ax.set_aspect('equal')

        mat = np.ma.array(mat, mask= mat==0.0)

        if edges_a is not None and edges_b is not None:
            c = ax.pcolorfast(edges_a, edges_b, mat)
            ax.invert_yaxis()
            ax.tick_params(top=True, labeltop=True)
        else:
            c = ax.matshow(mat)
        add_colorbar( c )

        path = os.path.join(str(tmp_path), f'eres_consistency_{plot_counter[0]:02}.png')
        plot_counter[0]+=1
        savefig(path, dpi=300)
        allure_attach_file(path)

    for key, Eres in eres_all.items():
        mat = Eres.matrix.FakeMatrix.data()

        if key=='a':
            name = 'AA (same)'
            e1 = edges_all[key]
            e2 = e1
        elif key=='a0':
            name = 'AA (bin center)'
            e1 = edges_all[key[0]]
            e2 = e1
        else:
            name = key.upper()
            e1 = edges_all[key[0]]
            e2 = edges_all[key[1]]

        plotmat(mat, name, e1, e2)

    def make_rebin_matrix_1d(
                             sizein: int,
                             ntogroup: Union[int, Tuple[int,...]],
                             average: bool=False,
                             weights: Optional[Tuple[float,...]]=None
    ) -> np.ndarray:
        if isinstance(ntogroup, int):
            sizeout = sizein // ntogroup
            ntogroup_combined = ntogroup
            ntogroup = (ntogroup, )
        else:
            ntogroup_combined = sum(ntogroup)
        sizeout = sizein // ntogroup_combined
        assert sizein%ntogroup_combined==0

        if average:
            if weights is None:
                Kblocks = [np.ones((ngroup, 1), dtype='i')/ngroup for ngroup in ntogroup]
            else:
                Kblocks = [np.vstack(weights) for ngroup in ntogroup]
        else:
            Kblocks = [np.ones((ngroup, 1), dtype='i') for ngroup in ntogroup]
        # print(f'{Kblocks=}')
        K = block_diag(*(Kblocks*sizeout))
        # print(f'{K=}')

        return K

    def make_rebin_matrices_2d(
                                shapein: Tuple[int, int],
                                ntogroup: Tuple[Union[int, Tuple[int,...]], Union[int, Tuple[int,...]]],
                                weights: Optional[Tuple[float,...]]=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sum over columns
        Average over rows
        """
        if isinstance(ntogroup, int):
            ntogroup = (ntogroup, ntogroup)
        Kleft  = make_rebin_matrix_1d(shapein[0], ntogroup[0]).T
        Kright = make_rebin_matrix_1d(shapein[1], ntogroup[1], average=True, weights=weights)

        return Kleft, Kright

    def rebinmat(mat: np.ndarray, with_k: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        left, right = with_k
        try:
            return np.matmul(left, np.matmul(mat, right))
        except ValueError:
            print('left:', left.shape)
            print('mat:', mat.shape)
            print('right:', right.shape)
            raise

    cnv_aa_ab = make_rebin_matrices_2d((nbins_a, nbins_a), (factors_ab, 1))
    cnv_aa_bb = make_rebin_matrices_2d((nbins_a, nbins_a), (factors_ab, factors_ab))
    cnv_aa_bc = make_rebin_matrices_2d((nbins_a, nbins_a), (factors_ac, factors_ab))
    cnv_aa_cb = make_rebin_matrices_2d((nbins_a, nbins_a), (factors_ab, factors_ac))
    cnv_ab_bb = make_rebin_matrices_2d((nbins_b, nbins_a), (1, factors_ab))
    cnv_aa_cc = make_rebin_matrices_2d((nbins_a, nbins_a), (factors_ac, factors_ac))
    cnv_bb_cc = make_rebin_matrices_2d((nbins_b, nbins_b), (factors_bc, factors_bc), weights=weights_ac)
    cnv_bc_cc = make_rebin_matrices_2d((nbins_c, nbins_b), (1, factors_bc), weights=weights_ac)
    cnv_cb_cc = make_rebin_matrices_2d((nbins_b, nbins_c), (factors_bc, 1))

    mat_a0 = Eres_a0.matrix.FakeMatrix.data()
    mat_a  = Eres_a.matrix.FakeMatrix.data()
    mat_aa0 = Eres_aa0.matrix.FakeMatrix.data()
    mat_aa = Eres_aa.matrix.FakeMatrix.data()
    mat_ab = Eres_ab.matrix.FakeMatrix.data()
    mat_bb = Eres_bb.matrix.FakeMatrix.data()
    mat_bc = Eres_bc.matrix.FakeMatrix.data()
    mat_cc = Eres_cc.matrix.FakeMatrix.data()
    mat_cb = Eres_cb.matrix.FakeMatrix.data()
    mat_ab_from_aa = rebinmat(mat_aa, cnv_aa_ab)
    mat_bb_from_aa = rebinmat(mat_aa, cnv_aa_bb)
    mat_bb_from_a0 = rebinmat(mat_a0, cnv_aa_bb)
    mat_bc_from_aa = rebinmat(mat_aa, cnv_aa_bc)
    mat_cb_from_aa = rebinmat(mat_aa, cnv_aa_cb)
    mat_bb_from_ab = rebinmat(mat_ab, cnv_ab_bb)
    mat_cc_from_aa = rebinmat(mat_aa, cnv_aa_cc)
    mat_cc_from_a0 = rebinmat(mat_a0, cnv_aa_cc)
    mat_cc_from_bb = rebinmat(mat_bb, cnv_bb_cc)
    mat_cc_from_bc = rebinmat(mat_bc, cnv_bc_cc)
    mat_cc_from_cb = rebinmat(mat_cb, cnv_cb_cc)

    def checkmat(key1: str, key2: str, mat1: np.ndarray, mat2: np.ndarray, *, check_assert=True) -> None:
        diff = mat1-mat2
        mdiff = np.fabs(diff).max()
        print(f'{key1.upper()} vs {key2.upper()} of {mat1.shape} max diff:', mdiff)

        if mdiff>1.e-8:
            plotmat(diff, f'{key1.upper()} vs {key2.upper()} diff: {mdiff}')

        if check_assert:
            assert mdiff<5.e9

    # Check the following conversions:
    # |---------|-----|------|----|----|----|----|----|----|
    # | From\To | AA0 |  AA  | AB | AC | BB | BC | CB | CC |
    # |:-------:|-----|:----:|:--:|:--:|:--:|:--:|----|----|
    # |    A0   | AA0 |      |    |    | BB |    |    | CC |
    # |    AA   |     | A0,A | AB |    | BB | BC | CB | CC |
    # |    AB   |     |      |    |    | BB |    |    |    |
    # |    BB   |     |      |    |    |    |    |    | CC |
    # |    CB   |     |      |    |    |    |    |    | CC |
    # |    BC   |     |      |    |    |    |    |    | CC |
    # |---------|-----|------|----|----|----|----|----|----|
    checkmat('a0', 'aa0', mat_a0, mat_aa0)
    checkmat('a', 'a0', mat_a, mat_a0, check_assert=False)
    checkmat('a', 'aa', mat_a, mat_aa)
    checkmat('ab', 'ab(aa)', mat_ab, mat_ab_from_aa)
    checkmat('bb', 'bb(aa)', mat_bb, mat_bb_from_aa)
    checkmat('bb', 'bb(a0)', mat_bb, mat_bb_from_a0)
    checkmat('bc', 'bc(aa)', mat_bc, mat_bc_from_aa)
    checkmat('bb', 'bb(ab)', mat_bb, mat_bb_from_ab)
    checkmat('cb', 'cb(aa)', mat_cb, mat_cb_from_aa)
    checkmat('cc', 'cc(aa)', mat_cc, mat_cc_from_aa)
    checkmat('cc', 'cc(bb)', mat_cc, mat_cc_from_bb)
    checkmat('cc', 'cc(a0)', mat_cc, mat_cc_from_a0)
    checkmat('cc', 'cc(bc)', mat_cc, mat_cc_from_bc)
    checkmat('cc', 'cc(cb)', mat_cc, mat_cc_from_cb)

    # plt.show()
    plt.close('all')

