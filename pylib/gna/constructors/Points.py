#!/usr/bin/env python

from load import ROOT as R
import numpy as np
from gna.converters import array_to_stdvector_size_t
from gna import context

"""Construct Points object from numpy array"""
def Points(array, *args, **kwargs):
    """Convert array to Points"""
    a = np.ascontiguousarray(array, dtype=context.current_precision_short())
    if len(a.shape)>2:
        raise Exception( 'Can convert only 1- and 2- dimensional arrays' )
    s = array_to_stdvector_size_t( a.shape )
    return R.GNA.GNAObjectTemplates.PointsT(context.current_precision())( a.ravel( order='F' ), s, *args, **kwargs )
