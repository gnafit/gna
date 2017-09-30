#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from converters import convert
from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument( '-t', '--triangular', action='store_true', help='force transformation to account for upper triangular matrix' )
opts = parser.parse_args()

env.defparameter( 'DiagScale',  central=1.0, relsigma=0.1 )

mat = N.matrix( N.arange(9.0).reshape(3,3) )
print( mat )
pmat = convert( mat, 'points' )

rd = R.RenormalizeDiag()
rd.renorm.inmat( pmat.points )

idt = R.Identity()
idt.identity.source( rd.renorm.outmat )

idt0 = R.Identity()
idt0.identity.source( pmat.points )

idt.identity.target.data()
idt0.identity.target.data()

print( 'Input' )
idt0.dump()

print( 'Output' )
idt.dump()
