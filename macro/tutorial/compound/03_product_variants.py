#!/usr/bin/env python

from load import ROOT as R
import gna.constructors as C
import numpy as np
# Create several points instances
n, n1, n2 = 12, 3, 4
points_list = [ C.Points(np.arange(i, i+n).reshape(n1, n2)) for i in range(5) ]

# Create product instances
tproduct_constructor = C.Product([p.points.points for p in points_list])
tproduct_multiply = C.Product()
tproduct_add_input = C.Product()

for i, p in enumerate(points_list):
    out = p.points.points
    tproduct_multiply.multiply(out)

    an_input = tproduct_add_input.add_input('input_{:02d}'.format(i))
    an_input(out)

# Print the structure of Product object
print('Product, configured via constructor')
tproduct_constructor.print()
print()

# Print the structure of Product object
print('Product, configured via multiply() method')
tproduct_multiply.print()
print()

# Print the structure of Product object
print('Product, configured via add_input() method')
tproduct_add_input.print()
print()


# Access the calcuation result:
print('Results:')
print(tproduct_constructor.product.product.data())
print()
print(tproduct_multiply.product.product.data())
print()
print(tproduct_add_input.product.product.data())

