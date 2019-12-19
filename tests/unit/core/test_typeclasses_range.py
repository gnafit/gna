#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.unittest import *
from gna import constructors as C
import numpy as np

def check_typeclasses_range(r, size, expect):
    what = list(r.vector(size, False))
    print('Check {}:'.format(size), what, expect)
    assert what==expect

def test_range_01():
    r = R.TypeClasses.Range(0)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0])
    check_typeclasses_range(r, 3, [0])

def test_range_02():
    r = R.TypeClasses.Range(1)
    r.dump(); print()

    check_typeclasses_range(r, 1, [])
    check_typeclasses_range(r, 2, [1])
    check_typeclasses_range(r, 3, [1])

def test_range_03():
    r = R.TypeClasses.Range(0,2)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0,1])
    check_typeclasses_range(r, 3, [0,1,2])
    check_typeclasses_range(r, 4, [0,1,2])

def test_range_04():
    r = R.TypeClasses.Range(0,-1)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0,1])
    check_typeclasses_range(r, 3, [0,1,2])
    check_typeclasses_range(r, 4, [0,1,2,3])

def test_range_05():
    r = R.TypeClasses.Range(2,-1)
    r.dump(); print()

    check_typeclasses_range(r, 1, [])
    check_typeclasses_range(r, 2, [])
    check_typeclasses_range(r, 3, [2])
    check_typeclasses_range(r, 4, [2,3])
    check_typeclasses_range(r, 5, [2,3,4])
    check_typeclasses_range(r, 6, [2,3,4,5])

def test_range_06():
    r = R.TypeClasses.Range(-3,-1)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0,1])
    check_typeclasses_range(r, 3, [0,1,2])
    check_typeclasses_range(r, 4, [1,2,3])
    check_typeclasses_range(r, 5, [2,3,4])

def test_range_07():
    r = R.TypeClasses.Range(-3,2)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0,1])
    check_typeclasses_range(r, 3, [0,1,2])
    check_typeclasses_range(r, 4, [1,2])
    check_typeclasses_range(r, 5, [2])
    check_typeclasses_range(r, 6, [])

def test_range_08():
    r = R.TypeClasses.Range(0, 5, 2)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0])
    check_typeclasses_range(r, 3, [0, 2])
    check_typeclasses_range(r, 4, [0, 2])
    check_typeclasses_range(r, 5, [0, 2, 4])
    check_typeclasses_range(r, 6, [0, 2, 4])

def test_range_09():
    r = R.TypeClasses.Range(0, 6, 2)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0])
    check_typeclasses_range(r, 3, [0, 2])
    check_typeclasses_range(r, 4, [0, 2])
    check_typeclasses_range(r, 5, [0, 2, 4])
    check_typeclasses_range(r, 6, [0, 2, 4])
    check_typeclasses_range(r, 7, [0, 2, 4, 6])
    check_typeclasses_range(r, 8, [0, 2, 4, 6])
    check_typeclasses_range(r, 9, [0, 2, 4, 6])
    check_typeclasses_range(r, 10, [0, 2, 4, 6])


def test_range_10():
    r = R.TypeClasses.Range(0, 5, 3)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0])
    check_typeclasses_range(r, 3, [0])
    check_typeclasses_range(r, 4, [0, 3])
    check_typeclasses_range(r, 5, [0, 3])
    check_typeclasses_range(r, 6, [0, 3])

def test_range_11():
    r = R.TypeClasses.Range(0, 6, 3)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0])
    check_typeclasses_range(r, 3, [0])
    check_typeclasses_range(r, 4, [0, 3])
    check_typeclasses_range(r, 5, [0, 3])
    check_typeclasses_range(r, 6, [0, 3])
    check_typeclasses_range(r, 7, [0, 3, 6])
    check_typeclasses_range(r, 8, [0, 3, 6])
    check_typeclasses_range(r, 9, [0, 3, 6])

def test_range_12():
    r = R.TypeClasses.Range(0, -1, 3)
    r.dump(); print()

    check_typeclasses_range(r, 1, [0])
    check_typeclasses_range(r, 2, [0])
    check_typeclasses_range(r, 3, [0])
    check_typeclasses_range(r, 4, [0, 3])
    check_typeclasses_range(r, 5, [0, 3])
    check_typeclasses_range(r, 6, [0, 3])
    check_typeclasses_range(r, 7, [0, 3, 6])
    check_typeclasses_range(r, 8, [0, 3, 6])
    check_typeclasses_range(r, 9, [0, 3, 6])
    check_typeclasses_range(r, 10, [0, 3, 6, 9])


if __name__ == "__main__":
    run_unittests(globals())

