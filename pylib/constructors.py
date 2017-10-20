#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define user constructors for C++ classes to simplify calling from python"""

from __future__ import print_function
from load import ROOT as R
import numpy as N

"""Construct Points object from numpy array"""
from converters import array_to_Points as Points

def Histogram( edges, data ):
    """Construct Histogram object from numpy arrays: edges and data"""
    if edges.size-1!=data.size:
        raise Exception( 'Bin edges and data are not consistent (%i and %i)'%( edges.size, data.size ) )

    return R.Histogram( data.size, edges, data )

