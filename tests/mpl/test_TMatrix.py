#!/usr/bin/env python

import ROOT as R
import numpy as N
from matplotlib import pyplot as P
from mpl_tools import bindings

m = N.arange(12,dtype='d').reshape(3,4)
print( 'Original' )
print(m)
print()

tm = R.TMatrixD( m.shape[0], m.shape[1], m.ravel() )
print( 'ROOT' )
tm.Print()
print()

print( 'Return' )
print(tm.get_buffer())

#
# matshow
#
tm.matshow( colorbar=True )

ax = P.gca()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'matshow' )

#
# matshow with mask
#
tm.matshow( colorbar=True, mask=0.0 )

ax = P.gca()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'matshow(mask=0.0)' )

#
# imshow with mask
#
fig = P.figure()
ax = P.subplot( 111 )
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'imshow (mask=0.0)' )
tm.imshow( colorbar=True, mask=0.0 )

P.show()
