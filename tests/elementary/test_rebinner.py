#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from converters import convert
from argparse import ArgumentParser
import constructors as C

edges   = N.array( [ 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8 ], dtype='d' )
edges_m = N.array( [ 0.1, 1.2,      3.4, 4.5,           7.8 ], dtype='d' )

ntrue = C.Histogram(edges, N.ones( edges.size-1 ) )
print(8)
rebin = R.Rebin( edges_m.size, edges_m, 3 )
print(9)
rebin.rebin.histin( ntrue )
print(10)

idy = R.Identity()
print(11)
idy.identity.source(rebin.rebin.histout)
print(12)

mat = convert(rebin.getDenseMatrix(), 'matrix')
print(13)
print( mat )

# print()
# print( (diff==0.0).all() and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )

# fig = P.figure()
# ax = P.subplot( 111 )
# ax.minorticks_on()
# ax.grid()
# ax.set_xlabel( 'Source bins' )
# ax.set_ylabel( 'Target bins' )
# ax.set_title( 'Bin edges scale conversion matrix' )

# c = ax.matshow( N.ma.array(mat, mask=mat==0.0), extent=[edges[0], edges[-1], edges[-1], edges[0]] )
# add_colorbar( c )

# P.show()
