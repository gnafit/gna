#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define converters to convert between Python (numpy) and C++ (std::vecotr, Eigen, etc) types
Implements individual converters along with generic 'convert' function."""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from collections import defaultdict
from inspect import getmro

debug_converters = True

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

    def msg( title ):
        res = title+"'{0}' to convert '{1}' ({2}) to '{3}'".format(
                      converter.__name__,
                      type(obj).__name__,
                      ', '.join([base.__name__ for base in bases]),
                      str(totype.__name__)
                    )
        if kwargs:
            res+=' [kwargs: %s]'%( str( kwargs ) )

        return res

    bases = list(getmro(type(obj)))
    for base in bases:
        bconverters = converters.get( base )
        if not bconverters:
            continue
        converter = bconverters.get( totype )
        if converter:
            break
    else:
        raise Exception(msg('Can not find converter'))

    if debug_converters:
        print( msg( 'Using converter' ) )
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

def stdvector_to_array( vector, dtype ):
    """Convert an std::vector to numpy.array"""
    return N.array( vector, dtype=dtype )

@save_converter( R.vector('int'), N.ndarray )
def stdvector_to_array_int( vector ):
    """Convert std::vector to array of int"""
    return stdvector_to_array( vector, 'i' )

@save_converter( R.vector('double'), N.ndarray )
def stdvector_to_array_double( vector ):
    """Convert std::vector to array of double"""
    return stdvector_to_array( vector, 'd' )

@save_converter( N.ndarray, R.Points )
def array_to_Points( array ):
    """Convert numpy array to Points"""
    if len(array.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    a = array.ravel( order='F' )
    s = array_to_stdvector_size_t( array.shape )
    return R.Points( a, s )

@save_converter( N.matrixlib.defmatrix.matrix, R.Points )
def matrix_to_Points( matrix ):
    """Convert numpy matrix to Points"""
    return array_to_Points( matrix.A )

