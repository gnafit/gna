#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_tools.helpers import savefig, add_colorbar
from argparse import ArgumentParser
import gna.constructors as C

mpl.rc('markers', fillstyle='none')

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-o', '--output', help='output file')
parser.add_argument('-s', '--show', action='store_true', help='output file')
parser.add_argument('--swap', action='store_true', help='swap direct and inverse')
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

coeff_a = 13
coeff_b = -24
fcn_direct = lambda e: (e*coeff_a + coeff_b)**0.5
fcn_inverse = lambda ep: (ep**2- coeff_b) / coeff_a

if opts.swap:
    fcn_direct, fcn_inverse = fcn_inverse, fcn_direct

orig      = np.array( [2.0, 3.0,  4.0,  5.0,  6.0,  7.0, 8.0, 9.0, 10.0] )
orig_proj = fcn_direct(orig)

mod_proj  = fcn_inverse(orig)

xfine = np.linspace(orig[0], orig[-1], 1000)
yfine = np.linspace(orig_proj[0], orig_proj[-1], 1000)

print('Edges original:', orig.shape, orig)
print('Edges original projected:', orig_proj.shape, orig_proj)
print('Edges modified projected back:', mod_proj.shape, mod_proj)

Orig = C.Points(orig)
Orig_proj = C.Points(orig_proj)
Mod_proj = C.Points(mod_proj)
ntrue = C.Histogram(orig, np.ones(orig.size-1))

nlB = C.HistNonlinearityB(R.GNA.DataPropagation.Propagate)
nlB.set(ntrue.hist, Orig_proj, Mod_proj)
nlB.add_input()

nlC = C.HistNonlinearityC(R.GNA.DataPropagation.Propagate)
nlC.set(ntrue.hist, Orig_proj, Mod_proj)
nlC.add_input()

nlC.printtransformations()

print('Get data')
mat_b = nlB.matrix.FakeMatrix.data()
mat_c = nlC.matrix.FakeMatrix.data()
edges_out = nlC.matrix.OutEdges.data()

print( 'Mat B and its sum' )
print( mat_b.shape )
print( mat_b )
matbsum_b = mat_b.sum( axis=0)
print( matbsum_b )
assert np.allclose(matbsum_b[1:-1], 1, rtol=0, atol=1.e-16)
# assert np.allclose(matbsum_c[1:-1], 1, rtol=0, atol=1.e-16)

print( 'Mat C and its sum' )
print( mat_c.shape )
print( mat_c )
matbsum_c = mat_c.sum( axis=0 )
print( matbsum_c )

print('Output edges')
print(edges_out.shape, edges_out)

def plot_matrix(mat, title: str, *, edges_a: np.ndarray, edges_b: np.ndarray=None) -> None:
    plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.set_xlabel( f'True energy, {edges_a.size-1} bins' )
    if edges_b is None:
        ax.set_ylabel( f'Modified energy, {edges_a.size-1} bins' )
    else:
        ax.set_ylabel( f'Modified energy, {edges_b.size-1} bins' )
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(edges_a[0], edges_a[-1])
    ax.set_ylim(edges_a[-1], edges_a[0])
    ax.grid()

    mmat = np.ma.array(mat, mask=mat==0.0)
    kwargs = {}

    if edges_b is None:
        kwargs['extent']=[edges_a[0], edges_a[-1], edges_a[-1], edges_a[0]]
        c = ax.matshow(mmat, **kwargs)

        ax.tick_params(labelbottom=True)
    else:
        # A, B = np.meshgrid(edges_a, edges_b, indexing='xy')
        c = ax.pcolorfast(edges_a, edges_b, mmat, **kwargs)
        # c = ax.pcolorfast((edges_a[0], edges_a[-1]), (edges_b[0], edges_b[-1]), mmat, **kwargs)

        ax.tick_params(top=True, labeltop=True)
        # ax.set_ylim(edges_b[-1], edges_b[0])


    add_colorbar(c)

    savefig(opts.output, suffix='matrix_0')

    ax.plot(xfine, fcn_direct(xfine), '--', color='white')
    savefig(opts.output, suffix='matrix_1')

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
plot_matrix(mat_b, 'Matrix B', edges_a=orig)
plot_matrix(mat_c, 'Matrix C', edges_a=orig, edges_b=edges_out)

if opts.show:
    plt.show()
