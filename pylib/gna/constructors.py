#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from converters import array_to_stdvector_size_t

"""Construct std::vector object from an array"""
from converters import list_to_stdvector as stdvector

def OutputDescriptors(outputs):
    descriptors=[]
    for output in outputs:
        if isinstance(output, R.OutputDescriptor):
            output = output
        elif isinstance(output, R.TransformationTypes.OutputHandle):
            output = R.OutputDescriptor(output)
        elif isinstance(output, R.SingleOutput):
            output=R.OutputDescriptor(output.single())
        else:
            raise Exception('Expect OutputHandle or SingleOutput object')
        descriptors.append(output)

    return stdvector(descriptors, 'OutputDescriptor*')

def wrap_constructor1(obj, dtype='d'):
    """Define a constructor for an object with signature Obje(size_t n, double*) with single array input"""
    def method(array, *args, **kwargs):
        array = N.ascontiguousarray(array, dtype=dtype)
        return R.SegmentWise(array.size, array, *args, **kwargs)
    return method

"""Construct VarArray object from vector of strings"""
def VarArray(varnames, *args, **kwargs):
    return R.VarArray(stdvector(varnames), *args, **kwargs)

"""Construct Points object from numpy array"""
def Points( array, *args, **kwargs ):
    """Convert array to Points"""
    a = N.ascontiguousarray(array, dtype='d')
    if len(a.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    s = array_to_stdvector_size_t( a.shape )
    return R.Points( a.ravel( order='F' ), s, *args, **kwargs )

"""Construct Sum object from list of SingleOutputs"""
def Sum(outputs=None, *args, **kwargs):
    if outputs is None:
        return R.Sum(*args, **kwargs)

    return R.Sum(OutputDescriptors(outputs), *args, **kwargs)

"""Construct WeightedSum object from lists of weights and input names/outputs"""
def WeightedSum(weights, inputs=None, *args, **kwargs):
    weights = stdvector(weights)
    if inputs is None:
        inputs=weights
    elif isinstance(inputs[0], str):
        inputs = stdvector(inputs)
    else:
        inputs = OutputDescriptors(inputs)

    return R.WeightedSum(weights, inputs, *args, **kwargs)

"""Construct Product object from list of SingleOutputs"""
def Product(outputs=None, *args, **kwargs):
    if outputs is None:
        return R.Product(*args, **kwargs)

    return R.Product(OutputDescriptors(outputs), *args, **kwargs)

"""Construct Bins object from numpy array"""
def Bins( array, *args, **kwargs ):
    """Convert array to Points"""
    a = N.ascontiguousarray(array, dtype='d')
    if len(a.shape)!=1:
        raise Exception( 'Edges should be 1d array' )
    return R.Bins( a, a.size-1, *args, **kwargs )

"""Construct Histogram object from two arrays: edges and data"""
def Histogram( edges, data=None, *args, **kwargs ):
    edges = N.ascontiguousarray(edges, dtype='d')
    reqsize = (edges.size-1)
    if data is None:
        data  = N.zeros(reqsize, dtype='d')
    else:
        if reqsize!=data.size:
            raise Exception( 'Bin edges and data are not consistent (%i and %i)'%( edges.size, data.size ) )
        data  = N.ascontiguousarray(data,  dtype='d')

    return R.Histogram( data.size, edges, data, *args, **kwargs )

"""Construct Histogram2d object from two arrays: edges and data"""
def Histogram2d( xedges, yedges, data=None, *args, **kwargs ):
    xedges = N.ascontiguousarray(xedges, dtype='d')
    yedges = N.ascontiguousarray(yedges, dtype='d')
    reqsize = (xedges.size-1)*(yedges.size-1)
    if data is None:
        data = N.zeros(reqsize, dtype='d')
    else:
        if reqsize!=data.size:
            raise Exception( 'Bin edges and data are not consistent (%i,%i and %i)'%( xedges.size, yedges.size, data.size ) )
        data = N.ascontiguousarray(data,   dtype='d').ravel(order='F')

    return R.Histogram2d( xedges.size-1, xedges, yedges.size-1, yedges, data, *args, **kwargs )

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
