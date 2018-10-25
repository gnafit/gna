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


mat = N.ones(10, dtype='d')

print( 'Input matrix (numpy)' )
print( mat )
print()

lmat = mat.ravel( order='F' )
shape = array_to_stdvector(mat.shape, 'size_t')

points = R.Points( lmat, shape )
points2 = R.Points( lmat, shape )

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


id3 = R.Identity()
id4 = R.Identity()
id3.identity.source( points2.points.points )
id4.identity.source(id3.identity.target)

id5 = R.Identity()
id5.identity.source(susu.sum.sum)

susu2 = R.Sum()
susu2.add(id5.identity.target)
susu2.add(id4.identity.target)


from gna.graphviz import GNADot
#kwargs=dict(
           # splines='ortho'
#           )
#ts = [ susu.sum ]
graph = GNADot( susu2.sum )
#graph = GNADot( id5.identity )
graph.write("dotfile.dot")

bres = susu2.data()
#bres = id5.data()


print( bres )
#IPython.embed()
#print( 'Result (C++ Data to numpy)' )
#print( points.points.points.data() )
#print( res )
#print( id2.identity.target.data() )
#print()

