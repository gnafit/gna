#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
from converters import *
from pprint import pprint

print( 'List converters' )
pprint(dict(converters))

print()
print( 'Test guessing' )
testlist = (
        N.array([0], dtype='d'),
        N.array([0], dtype='f'),
        N.array([0], dtype='i'),
        # N.array([0], dtype='u8'),
        [0.0],
        [0]
        )
for a in testlist:
    print( a, get_cpp_type( a ) )

print()
print( 'Test converter chain' )
d = N.arange(12, dtype='d')
chain = [
        ( R.vector('double'), dict() ),
        ( N.ndarray,          dict() ),
        ( R.vector('int'),    dict() ),
        ( N.ndarray,          dict() ),
        ]

def dump( r, s='' ):
    if s:
        print( s )
    print( '    ', type(r), type( r[0] ) )
    print( '    ', [v for v in r] )

r = d
dump( r, '  Initial' )
for t, o in chain:
    r = convert( r, t, debug=True, **o )
    dump( r )

print()
print( 'Test auto' )
for dtype in [ 'd', 'f', 'i' ]:
    a = N.arange(0.5, 12.5, dtype=dtype )
    print( 'From', a )
    c = convert( a, 'stdvector' )
    dump( c, 'to' )

print()
print( 'Test base conversion and Points' )
mat = N.matrix( N.arange(12, dtype='d').reshape(3, 4) )
p = convert( mat, R.Points, debug=True )
i = R.Identity()
i.identity.source( p )
print( 'mat' )
print( mat )
print( 'points' )
i.dump()

print()
print( 'Test Eigen' )
mat = N.matrix( N.arange(12, dtype='d').reshape(3, 4) )
em = convert( mat, R.Eigen.MatrixXd )
matb = convert( em, 'matrix' )
print( 'mat' )
print( mat )
print( 'eigen' )
R.EigenHelpers.dump( em )
print( 'mat back' )
print( mat )
print()

arr = N.array( N.arange(12, dtype='d').reshape(3, 4) )
em = convert( arr, R.Eigen.ArrayXXd )
arrb = convert( em, 'array' )
print( 'arr2' )
print( arr )
print( 'eigen' )
R.EigenHelpers.dump( em )
print( 'arr2 back' )
print( arrb )
print()

arr = N.array( N.arange(12, dtype='d') )
em = convert( arr, R.Eigen.ArrayXd )
arrb = convert( em, 'array' )
print( 'arr' )
print( arr )
print( 'eigen' )
R.EigenHelpers.dump( em )
print( 'arr back' )
print( arrb )

# mat = N.matrix( N.arange(12, dtype='d') ).T
# em = convert( mat, R.Eigen.VectorXd )
# print( 'vec' )
# print( mat )
# print( 'eigen' )
# R.EigenHelpers.dump( em )
# print()

