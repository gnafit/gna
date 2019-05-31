from __future__ import print_function
from gna import tutorial
import gna.constructors as C
import numpy as np
from gna.bindings import common



# Create numpy arrays for 1d and 2d cases
#narray1 = np.arange(5, 10)
#narray2 = np.arange(5, 20).reshape(5,3)
narray1 = [ 8.99E-05, 1.41E-04, 1.97E-04,2.97E-04,6.21E-04, 9.27E-04]
narray2 = [1.44E-04,1.49E-04,1.47E-04,1.69E-04,1.52E-04,1.72E-04]

x = [240, 500, 1000, 2000, 5000, 10000]
# Create a Points instances
parray1 = C.Points(narray1)
parray2 = C.Points(narray2)
px = C.Points(x)
print('Data 1d:',   parray1.points.points.data())
print('Data 2d:\n', parray2.points.points.data())

from matplotlib import pyplot as plt
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'N' )
ax.set_ylabel( 'T, c' )
ax.set_title( '' )

#px.points.points.plot_vs(parray1.points.points, '-o',label='CPU')
parray1.points.points.plot_vs(px.points.points, '-o',label='CPU')
parray2.points.points.plot_vs(px.points.points, '-s',label='GPU')
#parray1.points.points.plot('-o', label='CPU')
#parray2.points.points.plot('-s', label='GPU')

ax.legend(loc='upper left')
plt.show()
