from __future__ import print_function
from gna import tutorial
import gna.constructors as C
import numpy as np
from gna.bindings import common



# Create numpy arrays for 1d and 2d cases
#narray1 = np.arange(5, 10)
#narray2 = np.arange(5, 20).reshape(5,3)
#oscprob
#narray1 = [9.98E-05, 0.00014, 0.0002,  0.000296,  0.0005899, 0.001057]
#narray2 = [0.000158, 0.000159, 0.00016,  0.000159, 0.000172, 0.000187]

#exp
narray2 = [4.7180891037e-05, 5.12189865112e-05, 5.57267665863e-05, 5.84030151367e-05, 6.00388050079e-05, 6.11531734467e-05, 7.26218223572e-05,  7.4362039566e-05, 7.93287754059e-05]
narray1 = [1.93090438843E-05, 2.45640277863e-05,  3.00438404083e-05, 3.65960597992e-05, 4.3200969696e-05, 5.1506280899e-05, 6.53293132782e-05, 8.04049968719e-05, 9.06698703766e-05]

x = [100, 200,300, 400,500, 700,900, 1100, 1300]
# Create a Points instances
nar = [yy*1000000 for yy in narray1]
nar = np.array(nar)
nar = nar.T
nar2 = [yy*1000000 for yy in narray2]
nar2 = np.array(nar2)
nar2 = nar2.T
parray1 = C.Points(nar)
parray2 = C.Points(nar2)
px = C.Points(x)
print('Data 1d:',   parray1.points.points.data())
print('Data 2d:\n', parray2.points.points.data())

from matplotlib import pyplot as plt
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'N' )
ax.set_ylabel( 'T, mkc' )
ax.set_title( '' )

#px.points.points.plot_vs(parray1.points.points, '-o',label='CPU')
parray1.points.points.plot_vs(px.points.points, '-o',label='CPU')
parray2.points.points.plot_vs(px.points.points, '-s',label='GPU')
#parray1.points.points.plot('-o', label='CPU')
#parray2.points.points.plot('-s', label='GPU')

ax.legend(loc='upper left')
plt.show()
