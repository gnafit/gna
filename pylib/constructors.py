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

def Points( array, *args, **kwargs ):
    """Convert array to Points"""

    a = N.ascontiguousarray(array, dtype='d').ravel( order='F' )
    if len(a.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    s = array_to_stdvector_size_t( a.shape )
    return R.Points( a, s, *args, **kwargs )

def Sum(outputs=None, *args, **kwargs):
    """Construct Sum object from list of SingleOutputs"""
    if outputs is None:
        return R.Sum(*args, **kwargs)

    outputs = stdvector(outputs, 'OutputDescriptor*')
    return R.Sum(outputs, *args, **kwargs)

def Product(outputs=None, *args, **kwargs):
    """Construct Product object from list of SingleOutputs"""
    if outputs is None:
        return R.Product(*args, **kwargs)

    outputs = stdvector(outputs, 'OutputDescriptor*')
    return R.Product(outputs, *args, **kwargs)

def Bins( array, *args, **kwargs ):
    """Construct Bins object from numpy array"""

    a = N.ascontiguousarray(array, dtype='d')
    if len(a.shape)!=1:
        raise Exception( 'Edges should be 1d array' )
    return R.Bins( a, a.size-1, *args, **kwargs )

def Histogram( edges, data, *args, **kwargs ):
    """Construct Histogram object from two arrays: edges and data"""

    edges = N.ascontiguousarray(edges, dtype='d')
    data  = N.ascontiguousarray(data,  dtype='d')
    if (edges.size-1)!=data.size:
        raise Exception( 'Bin edges and data are not consistent (%i and %i)'%( edges.size, data.size ) )

    return R.Histogram( data.size, edges, data, *args, **kwargs )

def Histogram2d( xedges, yedges, data, *args, **kwargs ):
    """Construct Histogram2d object from two arrays: edges and data"""

    xedges = N.ascontiguousarray(xedges, dtype='d')
    yedges = N.ascontiguousarray(yedges, dtype='d')
    data   = N.ascontiguousarray(data,   dtype='d').ravel(order='F')
    if (xedges.size-1)*(yedges.size-1)!=data.size:
        raise Exception( 'Bin edges and data are not consistent (%i,%i and %i)'%( xedges.size, yedges.size, data.size ) )

    return R.Histogram2d( xedges.size-1, xedges, yedges.size-1, yedges, data, *args, **kwargs )

def GaussLegendre(edges, orders, *args, **kwargs):
    """Construct the GaussLegendre transformation based on bin edges and order(s)"""
    edges = N.ascontiguousarray(edges, dtype='d')
    if not isinstance(orders, int):
        orders = N.ascontiguousarray(orders, dtype='i')
    return R.GaussLegendre(edges, orders, edges.size-1, *args, **kwargs)

def Rebin( edges, rounding, *args, **kwargs ):
    """Construct Rebin object from array with edges"""
    if not isinstance( rounding, int ):
        raise Exception('Rebin rounding should be an integer')
    edges = N.ascontiguousarray(edges, dtype='d')
    return R.Rebin(edges.size, edges, int( rounding), *args, **kwargs )
