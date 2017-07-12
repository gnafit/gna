#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define converters to convert between Python (numpy) and C++ (std::vecotr, Eigen, etc) types
Implements individual converters along with generic 'convert' function."""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from collections import defaultdict
from inspect import getmro

# List all converters in dict: converters['from']['to']
converters = defaultdict( dict )

def convert(obj, totype, **kwargs):
    """Converto object obj to type totype.
    The converter is chosen from converters dictionary based on the type(obj) or one of it's base classes.

    :obj: object to convert
    :totype: the target type

    Example:
    convert( N.array([1, 2, 3]), R.vector('double') )
    convert( N.array([1, 2, 3]), R.vector, dtype='double' )
    """
    bases = [ type(obj) ]
    if hasattr(obj, '__bases__'):
        bases += getmro( obj )
    for base in bases:
        converter = converters.get( base ).get( totype )
        if converter: break
        break
    else:
        assert False, 'Can not find converter to convert %s (bases: %s) to %s'%(str(type(obj)), str(bases), str(totype))

    return converter( obj, **kwargs )

def save_converter( from_type, to_type ):
    """Make a decorator to store converter in a converters dictionary based on from/to types"""
    def decorator( converter ):
        converters[from_type][to_type] = converter
        return converter
    return decorator

@save_converter( N.ndarray, R.vector )
def array_to_stdvector( array, dtype ):
    """Convert an array to the std::vector<dtype>"""
    ret = R.vector(dtype)( len( array ) )
    for i, v in enumerate( array ):
        ret[i] = v
    return ret

@save_converter( N.ndarray, R.vector('double') )
def array_to_stdvector_double( array ):
    """Convert an array to the std::vector<double>"""
    return array_to_stdvector( array, 'double' )

@save_converter( N.ndarray, R.vector('size_t') )
def array_to_stdvector_size_t( array ):
    """Convert an array to the std::vector<size_t>"""
    return array_to_stdvector( array, 'size_t' )

@save_converter( N.ndarray, R.vector('int') )
def array_to_stdvector_int( array ):
    """Convert an array to the std::vector<int>"""
    return array_to_stdvector( array, 'int' )
