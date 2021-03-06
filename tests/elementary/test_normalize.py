#!/usr/bin/env python3

"""Check the Normalize transformation"""

import sys
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.constructors import Points
from gna.env import env

def example( s1, s2, step=1, start=None, length=None ):
    """Initialize data and transformations"""
    points = Points(N.arange(s1, s2, step, dtype='d'))
    if start is not None and length is not None:
        norm = R.Normalize(start, length)
    else:
        norm = R.Normalize()
    norm.normalize.inp( points.points.points )

    raw = points.points.points.data()
    data = norm.normalize.out.data()
    """
    Plot results
    """
    if "pytest" not in sys.modules:
        fig = P.figure()
        ax = P.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( '$i$' )
        ax.set_ylabel( 'height' )
        ax.set_title( 'Test normalize'+(start is not None and ' [%i, %i]=%i'%(start, start+length-1, length) or '') )

        ax.bar(range(len(data)),        raw,  width=0.5, label='raw' )
        ax.bar(N.arange(len(data))+0.5, data, width=0.5, label='normalized' )
        ax.legend(loc='upper left')

    if start is not None:
        print( 'Sum raw (%i:%i):'%(start, start+length), raw[start:start+length].sum() )
        res = data[start:start+length].sum()
        print( 'Sum data (%i:%i):'%(start, start+length), res, end=''  )
    else:
        print( 'Sum raw:', raw.sum() )
        res=data.sum()
        print( 'Sum data:', res, end='' )
    print( '    ', res==1.0 and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
    print()

def test_norm1():
    example( 1, 5 )

def test_norm2():
    example( 1, 5, start=0, length=4 )

def test_norm3():
    example( 1, 5, start=0, length=1 )

def test_norm4():
    example( 1, 5, start=1, length=1 )

def test_norm5():
    example( 1, 5, start=3, length=1 )

def test_norm6():
    example( 1, 20, 2 )

def test_norm7():
    example( 1, 20, 2, start=0, length=5 )

"""Exceptions"""
# test( 1, 5, start=-1, length=1 )
# test( 1, 5, start= 0, length=5 )
# test( 1, 5, start= 3, length=2 )
# test( 1, 5, start= 5, length=1 )

if __name__ == "__main__":
    test_norm1()
    test_norm2()
    test_norm3()
    test_norm4()
    test_norm5()
    test_norm6()
    test_norm7()
    if "pytest" not in sys.modules:
        P.show()
