#!/usr/bin/env python

from load import ROOT as R
from gna import constructors as C
import pytest

@pytest.mark.skip(reason="No way to effectively check for GPU-enabled configuration for now")
def test_01():
    arrays = ( [ [1.5, 1.5, 1.5], [1.5, 1.5, 1.5] ], [ [2.0, 2.0, 2.0], [2.0, 2.0, 2.0] ] )
    objects = [C.Points(a) for a in arrays]
    prod = C.Product(outputs=[p.single() for p in objects])
    prod.product.switchFunction('gpu')

    prod.print()
    print(prod.single().data())

    entry = R.OpenHandle(prod.product).getEntry()
    entry.functionargs.gpu.dump()

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')
