#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.converters import array_to_stdvector_size_t

"""Construct std::vector object from an array"""
from gna.converters import list_to_stdvector as stdvector

# Templates = R.GNA.GNAObjectTemplates
# print('templates', Templates)

_current_precision = 'double'
_current_precision_short = 'double'
def _set_current_precision(precision):
    global _current_precision, _current_precision_short
    assert precision in R.GNA.provided_precisions(), 'Unsupported precision '+precision
    _current_precision=precision
    _current_precision_short=precision[0]

class precision(object):
    """Context manager for the floating precision"""
    old_precision=''
    def __init__(self, precision):
        self.precision=precision

    def __enter__(self):
        self.old_precision = _current_precision
        _set_current_precision(self.precision)

    def __exit__(self, *args):
        _set_current_precision(self.old_precision)

class cuda(object):
    """Context manager for GPU
    Makes Initializer to switch transformations to "gpu" function after initialization"""
    backup_function=''
    def __init__(self):
        self.handle=R.TransformationTypes.InitializerBase

    def __enter__(self):
        self.backup_function = self.handle.getDefaultFunction()
        self.handle.setDefaultFunction('gpu')

    def __exit__(self, *args):
        self.handle.setDefaultFunction(self.backup_function)

def OutputDescriptors(outputs):
    descriptors=[]
    odescr = R.OutputDescriptorT(_current_precision, _current_precision)
    ohandle = R.TransformationTypes.OutputHandleT(_current_precision)
    singleoutput = R.SingleOutputT(_current_precision)
    for output in outputs:
        if isinstance(output, odescr):
            output = output
        elif isinstance(output, ohandle):
            output = odescr(output)
        elif isinstance(output, singleoutput):
            output=odescr(output.single())
        else:
            raise Exception('Expect OutputHandle or SingleOutput object')
        descriptors.append(output)

    return stdvector(descriptors, 'OutputDescriptorT<%s,%s>*'%(_current_precision,_current_precision))

def wrap_constructor1(obj, dtype='d'):
    """Define a constructor for an object with signature Obje(size_t n, double*) with single array input"""
    def method(array, *args, **kwargs):
        array = N.ascontiguousarray(array, dtype=dtype)
        return R.SegmentWise(array.size, array, *args, **kwargs)
    return method

"""Construct VarArray object from vector of strings"""
def VarArray(varnames, *args, **kwargs):
    return R.VarArray(stdvector(varnames), *args, **kwargs)

"""Construct Dummy object from vector of strings"""
def Dummy(shape, name, varnames, *args, **kwargs):
    return R.Dummy(shape, name, stdvector(varnames), *args, **kwargs)

"""Construct Points object from numpy array"""
def Points(array, *args, **kwargs):
    """Convert array to Points"""
    a = N.ascontiguousarray(array, dtype=_current_precision_short)
    if len(a.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    s = array_to_stdvector_size_t( a.shape )
    return R.GNA.GNAObjectTemplates.PointsT(_current_precision)( a.ravel( order='F' ), s, *args, **kwargs )

"""Construct Sum object from list of SingleOutputs"""
def Sum(outputs=None, *args, **kwargs):
    if outputs is None:
        return R.Sum(*args, **kwargs)

    return R.Sum(OutputDescriptors(outputs), *args, **kwargs)

"""Construct Sum object from list of SingleOutputs"""
def MultiSum(outputs=None, *args, **kwargs):
    cls = R.GNA.GNAObjectTemplates.MultiSumT(_current_precision)
    if outputs is None:
        return cls(*args, **kwargs)

    return cls(OutputDescriptors(outputs), *args, **kwargs)

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

"""Construct WeightedSumP object from lists outputs"""
def WeightedSumP(inputs, *args, **kwargs):
    outputs = OutputDescriptors(inputs)
    return R.WeightedSumP(outputs, *args, **kwargs)

"""Construct EnergyResolution object from lists of weights"""
def EnergyResolution(weights, *args, **kwargs):
    return R.EnergyResolution(stdvector(weights), *args, **kwargs)

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

def _wrap_integrator_1d(classname):
    def newfcn(edges, orders, *args, **kwargs):
        size = None
        if edges is not None:
            edges = N.ascontiguousarray(edges, dtype='d')
            size = edges.size-1
        if not isinstance(orders, int):
            orders = N.ascontiguousarray(orders, dtype='i')
            size = orders.size
        if size is None:
            raise Exception('Insufficient parameters to determine the number of bins')
        cls = getattr(R, classname)
        return cls(size, orders, edges, *args, **kwargs)
    return newfcn

IntegratorGL   = _wrap_integrator_1d('IntegratorGL')
IntegratorTrap = _wrap_integrator_1d('IntegratorTrap')
IntegratorRect = _wrap_integrator_1d('IntegratorRect')


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

