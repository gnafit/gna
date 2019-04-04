#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.converters import array_to_stdvector_size_t

"""Construct std::vector object from an array"""
from gna.converters import list_to_stdvector as stdvector
from gna import context

# Templates = R.GNA.GNAObjectTemplates
# print('templates', Templates)

def OutputDescriptors(outputs):
    descriptors=[]
    odescr = R.OutputDescriptorT(context.current_precision(), context.current_precision())
    ohandle = R.TransformationTypes.OutputHandleT(context.current_precision())
    singleoutput = R.SingleOutputT(context.current_precision())
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

    return stdvector(descriptors, 'OutputDescriptorT<%s,%s>*'%(context.current_precision(),context.current_precision()))

def wrap_constructor1(obj, dtype='d'):
    """Define a constructor for an object with signature Obje(size_t n, double*) with single array input"""
    def method(array, *args, **kwargs):
        array = N.ascontiguousarray(array, dtype=dtype)
        return R.SegmentWise(array.size, array, *args, **kwargs)
    return method

"""Construct VarArray object from vector of strings"""
def VarArray(vars, *args, **kwargs):
    cls = R.GNA.GNAObjectTemplates.VarArrayT(context.current_precision())
    ret = cls(stdvector(vars), *args, **kwargs)
    ret.transformations.front().updateTypes()
    return ret

"""Construct VarArrayPreallocated object from vector of strings"""
def VarArrayPreallocated(vars, *args, **kwargs):
    cls = R.GNA.GNAObjectTemplates.VarArrayPreallocatedT(context.current_precision())
    ret = cls(stdvector(vars), *args, **kwargs)
    return ret

def VarSum(varnames, *args, **kwargs):
    return R.GNA.GNAObjectTemplates.VarSumT(context.current_precision())(stdvector(varnames), *args, **kwargs)

def VarProduct(varnames, *args, **kwargs):
    return R.GNA.GNAObjectTemplates.VarProductT(context.current_precision())(stdvector(varnames), *args, **kwargs)

"""Construct Dummy object from vector of strings"""
def Dummy(shape, name, varnames, *args, **kwargs):
    return R.GNA.GNAObjectTemplates.DummyT(context.current_precision())(shape, name, stdvector(varnames), *args, **kwargs)

"""Construct Points object from numpy array"""
def Points(array, *args, **kwargs):
    """Convert array to Points"""
    a = N.ascontiguousarray(array, dtype=context.current_precision_short())
    if len(a.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    s = array_to_stdvector_size_t( a.shape )
    return R.GNA.GNAObjectTemplates.PointsT(context.current_precision())( a.ravel( order='F' ), s, *args, **kwargs )

"""Construct Identity object from list of SingleOutputs"""
def Identity(outputs=None, *args, **kwargs):
    if outputs is None:
        return Templates.IdentityT(current_precision)(*args, **kwargs)

    return Templates.IdentityT(current_precision)(OutputDescriptors(outputs), *args, **kwargs)

"""Construct Sum object from list of SingleOutputs"""
def Sum(outputs=None, *args, **kwargs):
    if outputs is None:
        return Templates.SumT(current_precision)(*args, **kwargs)

    return Templates.SumT(current_precision)(OutputDescriptors(outputs), *args, **kwargs)

"""Construct Sum object from list of SingleOutputs"""
def MultiSum(outputs=None, *args, **kwargs):
    cls = R.GNA.GNAObjectTemplates.MultiSumT(context.current_precision())
    if outputs is None:
        return cls(*args, **kwargs)

    return cls(OutputDescriptors(outputs), *args, **kwargs)

def PolyRatio(nominator=[], denominator=[], *args, **kwargs):
    nominator = stdvector(nominator, 'string')
    denominator = stdvector(denominator, 'string')
    return R.GNA.GNAObjectTemplates.PolyRatioT(context.current_precision())(nominator, denominator, *args, **kwargs)

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
        return Templates.ProductT(current_precision)(*args, **kwargs)

    return Templates.ProductT(current_precision)(OutputDescriptors(outputs), *args, **kwargs)

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

"""Construct Rebin object from array with edges"""
def Rebin( edges, rounding, *args, **kwargs ):
    if not isinstance( rounding, int ):
        raise Exception('Rebin rounding should be an integer')
    edges = N.ascontiguousarray(edges, dtype='d')
    return R.Rebin(edges.size, edges, int( rounding), *args, **kwargs )

def _wrap_parameter(classname):
    def newfcn(*args, **kwargs):
        template = getattr(R, classname)
        return template(context.current_precision())(*args, **kwargs)
    return newfcn

Variable              = _wrap_parameter('Variable')
Parameter             = _wrap_parameter('Parameter')
GaussianParameter     = _wrap_parameter('GaussianParameter')
UniformAngleParameter = _wrap_parameter('UniformAngleParameter')
ParameterWrapper      = _wrap_parameter('ParameterWrapper')

