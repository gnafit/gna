#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np

n, n1, n2 = 12, 3, 4

points_list = [ C.Points(np.arange(i, i+n).reshape(n1, n2)) for i in range(5) ]
tfactor = C.Points(np.linspace(0.5, 2.0, n).reshape(n1, n2))
tsum = C.Sum([p.points.points for p in points_list])
tprod = C.Product([tsum.sum.sum, tfactor.points.points])

for i, p in enumerate(points_list):
    p.points.setLabel('Sum input:\nP{:d}'.format(i))
tfactor.points.setLabel('Scale S')
tsum.sum.setLabel('Sum of matrices')
tprod.product.setLabel('Scaled matrix')
tprod.product.product.setLabel('result')

tsum.print()
print()

tprod.print()
print()

print('The sum:')
print(tsum.transformations[0].outputs[0].data())
print()

print('The scale:')
print(tfactor.transformations[0].outputs[0].data())
print()

print('The scaled sum:')
print(tprod.transformations[0].outputs[0].data())
print()

from gna.graphviz import savegraph
savegraph(tprod.transformations[0], 'output/07_graphviz.dot')
savegraph(tprod.transformations[0], 'output/07_graphviz.pdf')
savegraph(tprod.transformations[0], 'output/07_graphviz.png')
