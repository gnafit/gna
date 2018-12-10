#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R

def printall(*args):
    if args:
        t1a, t1b, t1c, t2a, t2b, t3a, t3b = args
        print("[{:^9s}]──────┐     ".format(t1a))
        print("            [{:^9s}]────[{:^9s}]".format(t2a, t3a))
        print("[{:^9s}]──────┤     ".format(t1b))
        print("            [{:^9s}]────[{:^9s}]".format(t2b, t3b))
        print("[{:^9s}]──────┘     ".format(t1c))
    else:
        printall('t1a', 't1b', 't1c', 't2a', 't2b', 't3a', 't3b')

def printflag(*args):
    printall(*(bool(arg) and 'tainted' or 'good' for arg in args))

sdict=['normal', 'frozen', 'frozen*']
def printstatus(*args):
    printall(*(sdict[arg.taintstatus()] for arg in args))

def printboth(*args):
    printall()
    print()
    printflag(*args)
    print()
    printstatus(*args)
    print()
    print()

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

    flags = [t1a, t1b, t1c, t2a, t2b, t3a, t3b]

    def reset():
        for t in flags:
            t.set(False)

    return flags, reset


def test1():
    flags, reset = make_flags()
    t1a, t1b, t1c, t2a, t2b, t3a, t3b = flags

    printboth(*flags)

    print('Reset')
    reset()
    printboth(*flags)

    print('Taint t1a')
    t1a.taint()
    printboth(*flags)
    reset()

    print('Taint t1c')
    t1c.taint()
    printboth(*flags)
    reset()

    print('Taint t1a and t1c')
    t1a.taint()
    t1c.taint()
    printboth(*flags)
    reset()

    print('Freeze t2b')
    t2b.freeze()
    printboth(*flags)
    reset()

    print('Taint t1c')
    t1c.taint()
    printboth(*flags)
    reset()

    print('Unfreeze t2b')
    t2b.unfreeze()
    printboth(*flags)
    reset()

    print('Try freezing tainted flag', end=': ')
    t1c.taint()
    res=R.GNAUnitTest.freeze(t1c)
    print(res and 'OK' or 'FAIL!')

def test2():
    flags, printall, reset = make_flags()
    t1a, t1b, t1c, t2a, t2b, t3a, t3b = flags

if __name__ == "__main__":
    test1()
