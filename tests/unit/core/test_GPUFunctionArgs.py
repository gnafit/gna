#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna import constructors as C

def test_01():
    arrays = ( [0.5], [ [1.5, 1.5, 1.5], [1.5, 1.5, 1.5] ], [ [2.0, 2.0, 2.0], [2.0, 2.0, 2.0] ] )
    objects = [C.Points(a) for a in arrays]
    prod = C.Product([p.single() for p in objects])

    prod.print()
    print(prod.single().data())

    entry = R.OpenHandle(prod.product).getEntry()
    entry.gpuargs.dump()


if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')
