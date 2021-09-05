#!/usr/bin/env python

from load import ROOT as R
import gna.constructors as C
import numpy as np
# Create several points instances
n, n1, n2 = 12, 3, 4
points_list = [ C.Points(np.arange(i, i+n).reshape(n1, n2)) for i in range(5) ]

# Create a sum instance
tsum = R.Sum()

for i, p in enumerate(points_list):
    out = p.points.points
    tsum.add(out)

    print('Input %i:'%i)
    print(out.data(), end='\n\n')

# Print the structure of Sum object
tsum.print()
print()


# Access the calcuation result:
print('Sum result:')
print(tsum.sum.sum.data())

