#!/usr/bin/env python

from load import ROOT as R
import gna.constructors as C
import numpy as np
# Create several points instances
n, n1, n2 = 12, 3, 4
points_list = [ C.Points(np.arange(i, i+n).reshape(n1, n2)) for i in range(5) ]

# Create sum instances
tsum_constructor = C.Sum([p.points.points for p in points_list])
tsum_add = R.Sum()
tsum_add_input = R.Sum()

for i, p in enumerate(points_list):
    out = p.points.points
    tsum_add.add(out)

    an_input = tsum_add_input.add_input('input_{:02d}'.format(i))
    an_input(out)

# Print the structure of Sum object
print('Sum, configured via constructor')
tsum_constructor.print()
print()

# Print the structure of Sum object
print('Sum, configured via add() method')
tsum_add.print()
print()

# Print the structure of Sum object
print('Sum, configured via add_input() method')
tsum_add_input.print()
print()


# Access the cacluation result:
print('Results:')
print(tsum_constructor.sum.sum.data())
print()
print(tsum_add.sum.sum.data())
print()
print(tsum_add_input.sum.sum.data())

