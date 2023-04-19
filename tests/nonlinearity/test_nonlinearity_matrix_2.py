#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.interpolate import interp1d
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
def fcn_direct(e):
    return (e*coeff_a + coeff_b)**0.5
def fcn_inverse(ep):
    return (ep**2- coeff_b) / coeff_a

if opts.swap:
    fcn_direct, fcn_inverse = fcn_inverse, fcn_direct

orig      = np.array( [2.0, 3.0,  4.0,  5.0,  6.0,  7.0, 8.0, 9.0, 10.0] )
orig_proj = fcn_direct(orig)

mod_proj  = fcn_inverse(orig)

xfine = np.linspace(orig[0], orig[-1], 1000)
yfine = np.linspace(orig_proj[0], orig_proj[-1], 1000)

print('Edges original:', orig)
print('Edges original projected:', orig_proj)
print('Edges modified projected back:', mod_proj)

Orig = C.Points(orig)
Orig_proj = C.Points(orig_proj)
Mod_proj = C.Points(mod_proj)
ntrue = C.Histogram(orig, np.ones(orig.size-1))

nlB = C.HistNonlinearityB(R.GNA.DataPropagation.Propagate)
nlB.set(ntrue.hist, Orig_proj, Mod_proj)
nlB.add_input()

print('Get data')
mat_b = nlB.matrix.FakeMatrix.data()

print( 'Mat B and its sum' )
print( mat_b )
matbsum = mat_b.sum( axis=0 )
print( matbsum )
assert np.allclose(matbsum[1:-1], 1, rtol=0, atol=1.e-16)

ntrue.hist.hist.data()
tflag1a = ntrue.hist.tainted()
tflag2a = nlB.matrix.tainted()
ntrue.hist.taint()
tflag1b = ntrue.hist.tainted()
tflag2b = nlB.matrix.tainted()
print('taints', tflag1a, tflag2a, tflag1b, tflag2b)
assert not tflag2b

def plot_matrix(mat):
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.set_xlabel( 'Source bins' )
    ax.set_ylabel( 'Target bins' )
    ax.set_title( 'Bin edges conversion matrix' )
    ax.set_aspect('equal')
    ax.set_xlim(orig[0], orig[-1])
    ax.set_ylim(orig[-1], orig[0])

    mmat = np.ma.array(mat, mask=mat==0.0)
    c = ax.matshow(mmat, extent=[orig[0], orig[-1], orig[-1], orig[0]])
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
plot_matrix(mat_b)

if opts.show:
    plt.show()
