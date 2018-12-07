#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R

def make_flags():
    t1a = R.taintflag()
    t1b = R.taintflag()
    t1c = R.taintflag()

    t2a = R.taintflag()
    t1a.subscribe(t2a)
    t1b.subscribe(t2a)

    t2b = R.taintflag()
    t1b.subscribe(t2b)
    t1c.subscribe(t2b)

    t3a = R.taintflag()
    t2a.subscribe(t3a)

    t3b = R.taintflag()
    t2b.subscribe(t3b)

    return t1a, t1b, t1c, t2a, t2b, t3a, t3b


def test1():
    flags = list(make_flags())
    t1a, t1b, t1c, t2a, t2b, t3a, t3b = flags

    def printall():
        print('t1a', bool(t1a), t1a.taintstatus(), end=', ')
        print('t1b', bool(t1b), t1b.taintstatus(), end=', ')
        print('t1c', bool(t1c), t1c.taintstatus())

        print('t2a', bool(t2a), t2a.taintstatus(), end=', ')
        print('t2b', bool(t2b), t2b.taintstatus())

        print('t3a', bool(t3a), t3a.taintstatus(), end=', ')
        print('t3b', bool(t3b), t3b.taintstatus())
        print()

    def reset():
        for t in flags:
            t.set(False)

    printall()

    print('Reset')
    reset()
    printall()

    print('Taint t1a')
    t1a.taint()
    printall()
    reset()

    print('Taint t1c')
    t1c.taint()
    printall()
    reset()

    print('Freeze t2b')
    t2b.freeze()
    printall()
    reset()

    print('Taint t1c')
    t1c.taint()
    printall()
    reset()

    print('Unfreeze t2b')
    t2b.unfreeze()
    printall()
    reset()

    print('Freeze t2b forever')
    t2b.freeze_forever()
    printall()
    reset()

    print('Unfreeze t2b')
    t2b.unfreeze()
    printall()
    reset()

if __name__ == "__main__":
    test1()
