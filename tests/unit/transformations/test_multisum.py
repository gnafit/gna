#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import run_unittests, makefloat
import numpy as np
from gna import constructors as C

def sum_arrays(arrays, lens):
    ret = []

    start=0
    for num in lens:
        end=start+num
        res=0.0
        for i in range(start, end):
            res+=arrays[i]
        ret.append(res)
        start=end

    return ret

def test_test():
    arrays = [np.arange(6).reshape(3,2)*i for i in range(12)]
    points = [C.Points(a) for a in arrays]
    outputs = [p.points.points for p in points]

    print('arrays')
    for i, arr in enumerate(arrays):
        print(arr, i)
    print()

    modes = [ [1], [2], [2, 1], [2, 3], [2, 3, 4], [3, 3, 4, 2] ]
    for mode in modes:
        print('mode', mode)
        ms = C.MultiSum()
        prev = 0
        for num in mode:
            last = prev+num
            if prev:
                ms.add_output()
            ms.add_inputs(C.OutputDescriptors(outputs[prev:last]))
            prev=last

        ms.print()

        checks = sum_arrays(arrays, mode)
        for it, trans in enumerate(ms.transformations.values()):
            for io, out in enumerate(trans.outputs.values()):
                data = out.data()

                check = checks[io]
                print(it, io)
                print('Result', data, sep='\n')
                print('Check', check, sep='\n')
                print('Datatype', out.datatype(), data.dtype)

                assert np.allclose(check, data)

        print()
        print()
        print()

makefloat('test_test', globals())

if __name__ == "__main__":
    run_unittests(globals())
