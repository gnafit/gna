#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
import numpy as N

"""Construct std::vector object from an array"""
from gna.converters import list_to_stdvector as stdvector
from gna import context

# Import constructors, defined in the submodules
from .Points import Points
from .Histogram import Histogram
from .Histogram import Histogram2d

def OutputDescriptors(outputs):
    descriptors=[]
    odescr = R.OutputDescriptorT(context.current_precision(), context.current_precision())
    ohandle = R.TransformationTypes.OutputHandleT(context.current_precision())
    singleoutput = R.SingleOutputT(context.current_precision())
    for output in outputs:
        if isinstance(output, odescr):
            append = output
        elif isinstance(output, ohandle):
            append = odescr(output)
        elif isinstance(output, singleoutput):
            append = odescr(output.single())
        else:
            raise Exception('Expect OutputHandle or SingleOutput object')
        descriptors.append(append)

    return stdvector(descriptors, 'OutputDescriptorT<%s,%s>'%(context.current_precision(), context.current_precision()))

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
def Dummy(shape, name, varnames=None, *args, **kwargs):
    if varnames:
        return R.GNA.GNAObjectTemplates.DummyT(context.current_precision())(shape, name, stdvector(varnames), *args, **kwargs)

    return R.GNA.GNAObjectTemplates.DummyT(context.current_precision())(shape, name, *args, **kwargs)

"""Construct Identity object from list of SingleOutputs"""
def Identity(outputs=None, **kwargs):
    if outputs is None:
        return R.GNA.GNAObjectTemplates.IdentityT(context.current_precision())(**kwargs)

    return R.GNA.GNAObjectTemplates.IdentityT(context.current_precision())(OutputDescriptors(outputs), **kwargs)

"""Construct Sum object from list of SingleOutputs"""
def Sum(outputs=None, **kwargs):
    if outputs is None:
        return R.GNA.GNAObjectTemplates.SumT(context.current_precision())(**kwargs)

    return R.GNA.GNAObjectTemplates.SumT(context.current_precision())(OutputDescriptors(outputs), **kwargs)

"""Construct Sum object from list of SingleOutputs"""
def MultiSum(outputs=None, **kwargs):
    cls = R.GNA.GNAObjectTemplates.MultiSumT(context.current_precision())
    if outputs is None:
        return cls(**kwargs)

    return cls(OutputDescriptors(outputs), **kwargs)

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

    return R.GNA.GNAObjectTemplates.WeightedSumT(context.current_precision())(weights, inputs, *args, **kwargs)

"""Construct WeightedSumP object from lists outputs"""
def WeightedSumP(inputs, *args, **kwargs):
    outputs = OutputDescriptors(inputs)
    return R.WeightedSumP(outputs, *args, **kwargs)

"""Construct EnergyResolution object from lists of weights"""
def EnergyResolution(weights, *args, **kwargs):
    return R.EnergyResolution(stdvector(weights), *args, **kwargs)

"""Construct SumBroadcast object from list of SingleOutputs"""
def SumBroadcast(outputs=None, **kwargs):
    if outputs is None:
        return R.SumBroadcast(**kwargs)

    return R.SumBroadcast(OutputDescriptors(outputs), **kwargs)

"""Construct Product object from list of SingleOutputs"""
def Product(outputs=None, **kwargs):
    if outputs is None:
        args=()
    else:
        args=OutputDescriptors(outputs),

    return R.GNA.GNAObjectTemplates.ProductT(context.current_precision())(*args, **kwargs)

"""Construct Product object from list of SingleOutputs"""
def ProductBC(outputs=None, **kwargs):
    if outputs is None:
        return R.GNA.GNAObjectTemplates.ProductBCT(context.current_precision())(**kwargs)

    return R.GNA.GNAObjectTemplates.ProductBCT(context.current_precision())(OutputDescriptors(outputs), **kwargs)

"""Construct Product object from list of SingleOutputs"""
def Exp(outputs=None, **kwargs):
    if outputs is None:
        return R.GNA.GNAObjectTemplates.ExpT(context.current_precision())(**kwargs)
    return R.GNA.GNAObjectTemplates.ExpT(context.current_precision())(OutputDescriptors(outputs), **kwargs)

"""Construct Bins object from numpy array"""
def Bins(array, *args, **kwargs):
    """Convert array to Points"""
    a = N.ascontiguousarray(array, dtype='d')
    if len(a.shape)!=1:
        raise Exception( 'Edges should be 1d array' )
    return R.Bins( a, a.size-1, *args, **kwargs )

def _wrap_integrator_1d(classname):
    def newfcn(edges, orders, *args, **kwargs):
        size = None
        if isinstance(edges, R.SingleOutput):
            edges_input, edges = edges, R.nullptr
            size = None
        elif edges is not None:
            edges = N.ascontiguousarray(edges, dtype='d')
            size = edges.size-1
            edges_input = None

        if not isinstance(orders, int):
            orders = N.ascontiguousarray(orders, dtype='i')
            size = orders.size

        cls = getattr(R, classname)

        if edges_input and isinstance(orders, int):
            ret = cls(orders, *args, **kwargs)
            edges_input >> ret.points.edges
            return ret

        if edges_input:
            size = edges_input.data().size

        if size is None:
            raise Exception('Insufficient parameters to determine the number of bins')

        ret=cls(size, orders, edges, *args, **kwargs)

        if edges_input:
            edges_input >> ret.points.edges

        return ret
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
