#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test random seed"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from converters import convert
import constructors as C

mu      = N.array( [ 100.0 ] )
mu_p    = C.Points(mu)
sigma_p = C.Points(mu**0.5)

toymc = R.NormalToyMC( False )
toymc.add( mu_p, sigma_p )

def test( n=11 ):
    res = N.zeros( shape=n, dtype='d' )
    for i in xrange( n ):
        toymc.nextSample()
        res[i] = toymc.toymc.data()
    return res

R.GNA.Random.seed( 1 )
toymc.reset()
r1 = test()

R.GNA.Random.seed( 1 )
toymc.reset()
r2 = test()

R.GNA.Random.seed( 1 )
r3 = test()

print( 'Sample 1 (explicit reset)' )
print( r1 )
print()

print( 'Sample 2 (explicit reset)' )
print( r2 )
print()

print( 'Sample 3 (implicit reset)' )
print( r3 )
print()

r12 = (r1==r2).all()
r32 = (r3==r2).all()

print( 'Equality: 1 and 2:', r12, r12 and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
print( r1==r2 )
print()

print( 'Equality: 2 and 3', r32, r32 and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
print( r3==r2 )
print()

