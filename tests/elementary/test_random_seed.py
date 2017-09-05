#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test random seed"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from converters import convert

mu      = N.array( [ 100.0 ] )
mu_p    = convert(mu, 'points')
sigma_p = convert(mu**0.5, 'points')

toymc = R.NormalToyMC( False )
toymc.add( mu_p, sigma_p )
out = R.Identity()
out.identity.source( toymc.toymc )

def test( n=11 ):
    res = N.zeros( shape=n, dtype='d' )
    for i in xrange( n ):
        toymc.nextSample()
        res[i] = out.identity.target.data()
    return res

R.GNA.Random.seed( 1 )
toymc.reset()
r1 = test()

R.GNA.Random.seed( 1 )
toymc.reset()
r2 = test()

R.GNA.Random.seed( 1 )
r3 = test()

print( 'Sample 1 (reset)' )
print( r1 )
print()

print( 'Sample 2 (reset)' )
print( r2 )
print()

print( 'Sample 3 (no reset)' )
print( r3 )
print()

r12 = (r1==r2).all()
r32 = (r3==r2).all()

print( 'Equality: 1 and 2:', r12, r12 and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
print( r1==r2 )
print()

print( 'Equality: 2 and 3', r32 )
print( r3==r2 )
print()

