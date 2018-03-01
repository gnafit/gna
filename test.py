#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test gpu"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
import IPython

def array_to_stdvector( array, dtype ):
    """Convert an array to the std::vector<dtype>"""
    ret = R.vector(dtype)( len( array ) )
    for i, v in enumerate( array ):
        ret[i] = v
    return ret


mat = N.ones(100, dtype='d')

print( 'Input matrix (numpy)' )
print( mat )
print()

lmat = mat.ravel( order='F' )
shape = array_to_stdvector(mat.shape, 'size_t')

points = R.Points( lmat, shape )

identity = R.Identity()
id2 = R.Identity()
#IPython.embed()
identity.identity.source( points.points.points )
id2.identity.source(identity.identity.target)
#res = id2.identity.target.data()


b_identity = R.Identity()
b_id2 = R.Identity()
b_identity.identity.source( points.points.points )
b_id2.identity.source(b_identity.identity.target)
#res = id2.identity.target.data()


susu = R.Sum()
susu.add(id2.identity.target)
susu.add(b_id2.identity.target)

bres = susu.data()
print( bres )

#IPython.embed()


print( 'Result (C++ Data to numpy)' )
#print( points.points.points.data() )
#print( res )
#print( id2.identity.target.data() )
print()

