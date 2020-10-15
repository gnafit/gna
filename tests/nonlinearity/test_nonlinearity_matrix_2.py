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

# matp_a = rescale_to_matrix_a( edges, edges_m, roundto=3 )
# matp_b = rescale_to_matrix_b( edges, centers_m, roundto=3 )
# matp_c = rescale_to_matrix_c( edges, edges_m, centers_m, roundto=3 )

Orig = C.Points(orig)
Orig_proj = C.Points(orig_proj)
Mod_proj = C.Points(mod_proj)
ntrue = C.Histogram(orig, np.ones(orig.size-1))

nlB = C.HistNonlinearityB(True)
nlB.set(ntrue.hist, Orig_proj, Mod_proj)
nlB.add_input()

print('Get data')
mat_b = nlB.matrix.FakeMatrix.data()

print( 'Mat B and its sum' )
print( mat_b )
print( mat_b.sum( axis=0 ) )

# print( 'Mat C and its sum' )
# print( matp_c )
# print( matp_c.sum( axis=0 ) )

# cmp(mat_a, matp_a)

# ntrue.hist.hist.data()
# tflag1a = ntrue.hist.tainted()
# tflag2a = nlA.matrix.tainted()
# ntrue.hist.taint()
# tflag1b = ntrue.hist.tainted()
# tflag2b = nlA.matrix.tainted()
# print('taints', tflag1a, tflag2a, tflag1b, tflag2b)
# assert not tflag2b

# def plot_edges_a(edges, centers, edges_m, centers_m):
    # fig = plt.figure()
    # ax = plt.subplot(111, xlabel='', ylabel='', title='Edges A')
    # ax.set_xlim(left=-2.0, right=10.0)
    # ax.set_ylim(0.0, 3.0)

    # ax.vlines(edges,   0.0, 5.0, linestyle='solid',  color='black', alpha=0.4, linewidth=0.2)
    # ax.vlines(edges,   0.0, 1.0, linestyle='solid',  color='blue', label='Original')
    # ax.vlines(centers, 0.0, 1.0, linestyle='dashed', color='blue')

    # ax.vlines(edges_m,   2.0, 3.0, linestyle='solid',  color='red', label='Modified')
    # ax.vlines(centers_m, 2.0, 3.0, linestyle='dashed', color='red')

    # for a, b in zip(edges, edges_m):
        # ax.plot( (a, b), (1.0, 2.0), color='gray' )

    # for i, c in enumerate(centers):
        # ax.text(c, 1.33, str(i), va='center', ha='center')

    # for i, c in enumerate(centers_m):
        # ax.text(c, 1.66, str(i), va='center', ha='center')

    # ax.legend(loc='upper left')

    # savefig(opts.output, suffix='edges_a')

# def plot_edges_b(edges, centers, edges_m, centers_m):
    # fig = plt.figure()
    # ax = plt.subplot(111, xlabel='', ylabel='', title='Edges B')
    # ax.set_xlim(left=-2.0, right=10.0)
    # ax.set_ylim(0.0, 3.0)

    # ax.vlines(edges,   0.0, 5.0, linestyle='solid',  color='black', alpha=0.4, linewidth=0.2)
    # ax.vlines(edges,   0.0, 1.0, linestyle='solid',  color='blue', label='Original')
    # ax.vlines(centers, 0.0, 1.0, linestyle='dashed', color='blue')

    # hwidths = 0.5*(edges[1:]-edges[:-1])
    # edges_m1 = centers_m-hwidths
    # edges_m2 = centers_m+hwidths
    # ax.vlines(edges_m1,  2.0, 2.5, linestyle='solid',  color='red', label='Left')
    # ax.vlines(edges_m2,  2.5, 3.0, linestyle='solid',  color='magenta', label='Right')
    # ax.vlines(centers_m, 2.0, 3.0, linestyle='dashed', color='red')

    # for a, b in zip(edges, edges_m):
        # ax.plot( (a, b), (1.0, 2.0), color='gray' )

    # for i, c in enumerate(centers):
        # ax.text(c, 1.33, str(i), va='center', ha='center')

    # for i, c in enumerate(centers_m):
        # ax.text(c, 1.66, str(i), va='center', ha='center')

    # ax.legend(loc='upper left')

    # savefig(opts.output, suffix='edges_b')

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

# def plot_edges():
    # fig = plt.figure()
    # ax = plt.subplot(111, xlabel='', ylabel='', title='')
    # ax.minorticks_on()
    # ax.grid()

    # ax.plot(edges[1:],      edges_m[1:], 'o-', label='direct')
    # ax.plot(edges_back[1:], edges[1:],   'o-', label='inverse')
    # ax.plot(edges[1:],      edges_back[1:], 'o-', label='inverse')

    # ax.legend()

    # fig = plt.figure()
    # ax = plt.subplot(111, xlabel='', ylabel='', title='')
    # ax.minorticks_on()
    # # ax.grid()

    # ax.vlines(edges[1:],      0.0, 1.0, linestyle='solid', label='Original')
    # ax.vlines(edges_m[1:],    1.0, 2.0, linestyle='solid', label='Modified')
    # ax.vlines(edges_back[1:], 2.0, 3.0, linestyle='solid', label='Modified')

    # ax.legend()

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

# plot_edges_a(edges, centers, edges_m, centers_m)
# plot_edges_b(edges, centers, edges_m, centers_m)

plot_matrix(mat_b)

# plot_edges()

if opts.show:
    plt.show()
