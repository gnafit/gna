#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from converters import array_to_stdvector_size_t

"""Construct std::vector object from an array"""
from converters import list_to_stdvector as stdvector

def wrap_constructor1(obj, dtype='d'):
    """Define a constructor for an object with signature Obje(size_t n, double*) with single array input"""
    def method(array, *args, **kwargs):
        array = N.ascontiguousarray(array, dtype=dtype)
        return R.SegmentWise(array.size, array, *args, **kwargs)
    return method

"""Construct Points object from numpy array"""
def Points( array, *args, **kwargs ):
    """Convert array to Points"""
    if len(array.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    a = N.ascontiguousarray(array, dtype='d').ravel( order='F' )
    s = array_to_stdvector_size_t( array.shape )
    return R.Points( a, s, *args, **kwargs )

"""Construct Histogram object from two arrays: edges and data"""
def Histogram( edges, data, *args, **kwargs ):
    edges = N.ascontiguousarray(edges, dtype='d')
    data  = N.ascontiguousarray(data,  dtype='d')
    if (edges.size-1)!=data.size:
        raise Exception( 'Bin edges and data are not consistent (%i and %i)'%( edges.size, data.size ) )

    return R.Histogram( data.size, edges, data, *args, **kwargs )

"""Construct the GaussLegendre transformation based on bin edges and order(s)"""
def GaussLegendre(edges, orders, *args, **kwargs):
    edges = N.ascontiguousarray(edges, dtype='d')
    if not isinstance(orders, int):
        orders = N.ascontiguousarray(orders, dtype='i')
    return R.GaussLegendre(edges, orders, edges.size-1, *args, **kwargs)

"""Construct Rebin object from array with edges"""
def Rebin( edges, rounding, *args, **kwargs ):
    if not isinstance( rounding, int ):
        raise Exception('Rebin rounding should be an integer')
    edges = N.ascontiguousarray(edges, dtype='d')
    return R.Rebin(edges.size, edges, int( rounding), *args, **kwargs )
