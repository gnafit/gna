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

# Add nicknames for types
nicknames = {
        R.vector:  'stdvector',
        N.ndarray: 'array',
        N.matrixlib.defmatrix.matrix: 'matrix'
        }

def convert(obj, totype, debug=False, **kwargs):
    """Converto object obj to type totype.
    The converter is chosen from converters dictionary based on the type(obj) or one of it's base classes.

    :obj: object to convert
    :totype: the target type

    Order:
      1. Set type to type(obj).
      2. Try to find converter for the current type. Return if found.
      3. Try to find 'base' converter for the current type. Convert obj to base and return convert(newobj) if 'base' converter found.
      4. Set type to next base type of obj. Repeat from 2.

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

    bases = getmro(type(obj))
    for base in bases:
        bconverters = converters.get( base )
        if not bconverters:
            continue
        converter = bconverters.get( totype )
        if converter:
            break

        if 'base' in bconverters:
            if debug:
                print( 'Convert', type(obj).__name__, 'to base' )
            return convert( bconverters['base'](obj), totype, debug, **kwargs )
    else:
        raise Exception(msg('Can not find converter'))

    if debug:
        print( msg( 'Using converter' ) )
    return converter( obj, **kwargs )

def save_converter( from_type, to_type ):
    """Make a decorator to store converter in a converters dictionary based on from/to types"""
    def decorator( converter ):
        fts, tts = [from_type], [to_type]
        if from_type in nicknames:
            fts.append( nicknames[from_type] )
        if to_type in nicknames:
            tts.append( nicknames[to_type] )
        for ft in fts:
            for tt in tts:
                converters[ft][tt] = converter
        return converter
    return decorator

def get_cpp_type( array ):
    """Guess appropriate C++ type to store data based on array.dtype or type"""
    if hasattr( array, 'dtype' ):
        typemap = {
                'int32':   'int',
                'float64': 'double',
                'float32': 'float',
                # 'uint64':  'size_t',
                }
        atype = array.dtype.name
    else:
        typemap = {
                int: 'int',
                float: 'double'
                }
        atype = type( array[0] )
    ret = typemap.get( atype )
    if not ret:
        raise Exception( 'Do not know how to convert type '+atype )
    return ret

@save_converter( N.ndarray, R.vector )
def array_to_stdvector( array, dtype='auto' ):
    """Convert an array to the std::vector<dtype>"""
    if dtype=='auto':
        dtype = get_cpp_type( array )
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

@save_converter( R.vector('size_t'), N.ndarray )
def stdvector_to_array_double( vector ):
    """Convert std::vector to array of double"""
    return stdvector_to_array( vector, 'u8' )

@save_converter( N.matrixlib.defmatrix.matrix, 'base' )
def matrix_to_array( matrix ):
    """Convert numpy matrix to array"""
    return matrix.A

@save_converter( N.ndarray, R.Points )
def array_to_Points( array ):
    """Convert numpy array to Points"""
    if len(array.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    a = array.ravel( order='F' )
    s = array_to_stdvector_size_t( array.shape )
    return R.Points( a, s )

Datadouble = R.Data('double')
VectorXd, MatrixXd, ArrayXd, ArrayXXd = Datadouble.VectorXT, Datadouble.MatrixXT, Datadouble.ArrayXT, Datadouble.ArrayXXT
import IPython
IPython.embed()
@save_converter( N.ndarray,  )
def array_to_Points( array ):
    """Convert numpy array to Points"""
    if len(array.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    a = array.ravel( order='F' )
    s = array_to_stdvector_size_t( array.shape )
    return R.Points( a, s )

