#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.interpolate import interp1d
from argparse import ArgumentParser
import gna.constructors as C
from scipy.linalg import block_diag

mpl.rc('markers', fillstyle='none')

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-o', '--output', help='output file')
parser.add_argument('-s', '--show', action='store_true', help='output file')
parser.add_argument('--swap', action='store_true', help='swap direct and inverse')
opts = parser.parse_args()

def make_rebin_matrix_1d(sizein, ntogroup):
    sizeout = sizein//ntogroup
    assert sizein%ntogroup ==0

    Kblock = np.ones((ntogroup,1), dtype='i')
    # print(f'Kblock: {Kblock}')
    K = block_diag(*[Kblock]*sizeout)

    return K

def make_rebin_matrices_2d(shapein, ntogroup):
    """
    Sum over columns
    Average over rows
    """
    if isinstance(ntogroup, int):
        ntogroup = (ntogroup, ntogroup)
    Kleft  = make_rebin_matrix_1d(shapein[0], ntogroup[0]).T
    Kright = make_rebin_matrix_1d(shapein[1], ntogroup[1])
    Kright = Kright/float(ntogroup[1])

    return Kleft, Kright

def reduce_matrix(mat, n):
    Kleft, Kright = make_rebin_matrices_2d(mat.shape, n)
    return np.matmul(np.matmul(Kleft, mat), Kright)

def plot_matrix(mat, curve_x=None, curve_y=None, reshapen=None):
    nbins = mat.shape[0]
    title =  'Bin edges conversion matrix {}x{}'.format(nbins, nbins)
    suffix = 'matrix_{}'.format(nbins)

    if reshapen:
        nbins_orig, nbins = nbins, nbins//reshapen
        title =  'Bin edges conversion matrix {}x{} from {}x{}'.format(nbins, nbins, nbins_orig, nbins_orig)
        suffix = 'matrix_{}_{}'.format(nbins, nbins_orig)

        mat = reduce_matrix(mat, reshapen)

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.set_xlabel( 'Source bins' )
    ax.set_ylabel( 'Target bins' )
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(orig[0], orig[-1])
    ax.set_ylim(orig[-1], orig[0])

    mmat = np.ma.array(mat, mask=mat==0.0)
    c = ax.matshow(mmat, extent=[orig[0], orig[-1], orig[-1], orig[0]])
    add_colorbar(c)

    savefig(opts.output, suffix=suffix)

    if curve_x is None or curve_y is None:
        return

    ax.plot(curve_x, curve_y, '--', color='white')
    savefig(opts.output, suffix='{}_c'.format(suffix))

def cmp(a, b):
    good = np.allclose(a, b, rtol=0, atol=1e-15)
    print( good and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
    if not good:
        diff = a-b
        print( 'diff' )
        print( diff )
        print()
        print(diff)

coeff_a = 13
coeff_b = -24
def fcn_direct(e):
    return (e*coeff_a + coeff_b)**0.5
def fcn_inverse(ep):
    return (ep**2- coeff_b) / coeff_a

if opts.swap:
    fcn_direct, fcn_inverse = fcn_inverse, fcn_direct

e1, e2 = 2.0, 10.0
xfine = np.linspace(e1, e2, 1000)
fxfine = fcn_direct(xfine)
matrices = {}
for step in (1.0, 0.5, 0.1):
    orig = np.arange(e1, e2+1.e-9, step)

    orig_proj = fcn_direct(orig)
    mod_proj  = fcn_inverse(orig)

    yfine = np.linspace(orig_proj[0], orig_proj[-1], 1000)

    print('Edges original:', orig)
    print('Edges original projected:', orig_proj)
    print('Edges modified projected back:', mod_proj)

    Orig = C.Points(orig)
    Orig_proj = C.Points(orig_proj)
    Mod_proj = C.Points(mod_proj)
    ntrue = C.Histogram(orig, np.ones(orig.size-1))

    nlB = C.HistNonlinearityB(True)
    nlB.set(ntrue.hist, Orig_proj, Mod_proj)
    nlB.add_input()

    print('Get data')
    mat_b = nlB.matrix.FakeMatrix.data()
    nbins = mat_b.shape[0]
    matrices[nbins] = mat_b.copy()

    print( 'Mat B and its sum' )
    print( mat_b )
    matbsum = mat_b.sum( axis=0 )
    print( matbsum )
    # checkones = matbsum!=0.0
    # assert np.allclose(matbsum[checkones][1:-1], 1, rtol=0, atol=1.e-16)

    ntrue.hist.hist.data()
    tflag1a = ntrue.hist.tainted()
    tflag2a = nlB.matrix.tainted()
    ntrue.hist.taint()
    tflag1b = ntrue.hist.tainted()
    tflag2b = nlB.matrix.tainted()
    print('taints', tflag1a, tflag2a, tflag1b, tflag2b)
    assert not tflag2b

    def plot_curve():
        fig = plt.figure()
        ax = plt.subplot(111, xlabel='X original', ylabel='X modified', title='')
        ax.minorticks_on()
        ax.grid()

        c = ax.plot(orig, orig_proj, 'o', label='Direct')[0].get_color()
        ax.plot(xfine, fcn_direct(xfine), '--', color=c)

        c = ax.plot(mod_proj, orig, 'o', label='Inverse')[0].get_color()
        ax.plot(fcn_inverse(yfine), yfine, '--', color=c)

        ax.legend()

    plot_curve()

Nbins = matrices.keys()
for nbins, mat in matrices.items():
    plot_matrix(mat, xfine, fxfine)

    for nbins1 in Nbins:
        if nbins1==nbins:
            continue

        ratio = nbins//nbins1
        if nbins%nbins1:
            continue

        plot_matrix(mat, xfine, fxfine, reshapen=ratio)

if opts.show:
    plt.show()
