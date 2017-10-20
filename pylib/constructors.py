#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from load import ROOT as R
import numpy as N
from converters import array_to_stdvector_size_t

"""Construct Points object from numpy array"""
def Points( array ):
    """Convert numpy array to Points"""
    if len(array.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    a = N.asarray(array).ravel( order='F' )
    s = array_to_stdvector_size_t( array.shape )
    return R.Points( a, s )

def Histogram( edges, data ):
    """Construct Histogram object from numpy arrays: edges and data"""
    if edges.size-1!=data.size:
        raise Exception( 'Bin edges and data are not consistent (%i and %i)'%( edges.size, data.size ) )

    return R.Histogram( data.size, edges, data )

