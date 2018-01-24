#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from converters import convert
from argparse import ArgumentParser
import constructors as C

parser = ArgumentParser()
parser.add_argument( '-v', '--value', default=1.0, type=float, help='renorm value' )
parser.add_argument( '-n', '--ndiag', default=1, type=int, help='number of diagonals' )
parser.add_argument( '-u', '--upper', action='store_true', help='force transformation to account for upper triangular matrix' )
parser.add_argument( '-o', '--offdiag', action='store_true', help='force transformation to account for upper triangular matrix' )
opts = parser.parse_args()

env.defparameter( 'DiagScale',  central=opts.value, relsigma=0.1 )

mat = N.matrix( N.arange(16.0).reshape(4,4) )
print( 'Raw matrix' )
print( mat )

print( 'Sum over rows' )

print( mat.A.sum(axis=0) )
print( 'Raw normalized matrix' )
print( mat.A/mat.A.sum(axis=0) )

pmat = C.Points( mat )

rd = R.RenormalizeDiag( opts.ndiag, int(opts.offdiag), int(opts.upper) )
rd.renorm.inmat( pmat.points )

idt = R.Identity()
idt.identity.source( rd.renorm.outmat )

idt0 = R.Identity()
idt0.identity.source( pmat.points )

idt.identity.target.data()
idt0.identity.target.data()

if opts.upper:
    print( 'Upper triangle mode' )

print( 'Input (Eigen)' )
idt0.dump()

print( 'Output (Eigen)' )
idt.dump()

print( 'Output (python)' )
m = N.matrix(idt.identity.target.data())
print( m )
print( 'Output colsum (python)' )
print( m.sum( axis=0 ) )
