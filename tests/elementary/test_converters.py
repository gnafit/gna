#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna.converters import *
from pprint import pprint

print( 'List converters' )
pprint(dict(converters))
print()

print( 'Try exception' )
d = N.array([0])
try:
    convert( d, object, debug=True )
except Exception as e:
    print( 'Planned exception:', e )

try:
    convert( object(), object, debug=True )
except Exception as e:
    print( 'Planned exception:', e )
print()

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
    c = convert( a, 'stdvector', debug=True  )
    dump( c, 'to' )
    b = convert( c, 'array', debug=True  )
    dump( b, 'back' )

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
mat = N.matrix( N.arange(1, 13, dtype='d').reshape(3, 4) )
em = convert( mat, R.Eigen.MatrixXd, debug=True  )
matb = convert( em, 'matrix', debug=True  )
print( 'matrix' )
print( mat )
print( 'MatrixXd' )
#  R.EigenHelpers.dump( em )
print( 'matrix back' )
print( mat )
print()

arr = N.array( N.arange(2, 14, dtype='d').reshape(3, 4) )
em = convert( arr, R.Eigen.ArrayXXd, debug=True  )
arrb = convert( em, 'array', debug=True  )
print( 'array 2d' )
print( arr )
print( 'ArrayXXd' )
#  R.EigenHelpers.dump( em )
print( 'arr2 back' )
print( arrb )
print()

arr = N.array( N.arange(3, 15, dtype='d') )
em = convert( arr, R.Eigen.ArrayXd, debug=True  )
arrb = convert( em, 'array', debug=True  )
print( 'array' )
print( arr )
print( 'ArrayXd' )
#  R.EigenHelpers.dump( em )
print( 'arr back' )
print( arrb )

mat = N.matrix( N.arange(4, 16, dtype='d') ).T
em = convert( mat, R.Eigen.VectorXd, debug=True  )
matb = convert( em, 'matrix', debug=True  )
print( 'vector', mat.shape[0], mat.shape[1] )
print( mat )
print( 'VectorXd', em.rows(), em.cols() )
#  R.EigenHelpers.dump( em )
print( 'vec back' )
print( matb )
print()

arr = N.array( N.arange(2, 14, dtype='f').reshape(3,4) )
tm = convert( arr, R.TMatrixD, debug=True  )
tmf = convert( arr, R.TMatrixF, debug=True  )
tm1 = convert( arr, 'tmatrix', debug=True  )
arrb = convert( tm, 'array', debug=True  )
arrbf = convert( tmf, 'array', debug=True  )
print( 'array' )
print( arr )
print( 'TMatrixD' )
tm.Print()
print( 'TMatrixF' )
tm.Print()
print( 'TMatrix*' )
print(type(tm1))
tm1.Print()
print( 'arr back' )
print( arrb )
print( arrbf )

arr = [ 'a', 'b', 'cde' ]
vec = convert( arr, 'stdvector' )
print( arr )
print( [s for s in vec] )
