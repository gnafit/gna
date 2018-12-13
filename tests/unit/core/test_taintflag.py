#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R

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
    args=list(bool(arg) and 'tainted' or 'good' for arg in args)
    printall(*args)
    return args

sdict=['normal', 'frozen', 'frozen*', 'pass']
def printstatus(*args):
    args=list(sdict[arg.taintstatus()] for arg in args)
    printall(*args)
    return args

def printboth(*args, **kwargs):
    printall()
    print()
    args1=printflag(*args)
    print()
    args2=printstatus(*args)
    print()
    print()

    flags=kwargs.pop('flags', [])
    if flags:
        if args1!=flags:
            print('args', args1)
            print('flags', flags)
            raise Exception('Invalid flags')

    statuses=kwargs.pop('statuses', [])
    if statuses:
        if args2!=statuses:
            print('args', args2)
            print('flags', statuses)
            raise Exception('Invalid statuses')


def test_taintflag_01():
    printboth(*flags, flags=['tainted', 'tainted', 'tainted', 'tainted', 'tainted', 'tainted', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_02():
    print('Reset')
    reset()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_03():
    print('Taint t1a')
    t1a.taint()
    printboth(*flags, flags=['tainted', 'good', 'good', 'tainted', 'good', 'tainted', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_04():
    print('Taint t1c')
    reset()
    t1c.taint()
    printboth(*flags, flags=['good', 'good', 'tainted', 'good', 'tainted', 'good', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_05():
    print('Taint t1a and t1c')
    reset()
    t1a.taint()
    t1c.taint()
    printboth(*flags, flags=['tainted', 'good', 'tainted', 'tainted', 'tainted', 'tainted', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_06():
    print('Taint t1a and t2a')
    reset()
    t1a.taint()
    t2a.taint()
    printboth(*flags, flags=['tainted', 'good', 'good', 'tainted', 'good', 'tainted', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_07():
    print('Freeze t2b')
    reset()
    t2b.freeze()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'frozen', 'normal', 'normal'])

def test_taintflag_08():
    print('Taint t1c')
    reset()
    t1c.taint()
    printboth(*flags, flags=['good', 'good', 'tainted', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'frozen*', 'normal', 'normal'])

def test_taintflag_09():
    print('Unfreeze t2b')
    t2b.unfreeze()
    printboth(*flags, flags=['good', 'good', 'tainted', 'good', 'tainted', 'good', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

def test_taintflag_10():
    print('Set t2b pass through')
    reset()
    t2b.set_pass_through()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'pass', 'normal', 'normal'])

def test_taintflag_11():
    print('Taint t1b')
    reset()
    t1b.taint()
    printboth(*flags, flags=['good', 'tainted', 'good', 'tainted', 'tainted', 'tainted', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'pass', 'normal', 'normal'])

def test_taintflag_12():
    print('Taint t2b')
    reset()
    t2b.taint()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'tainted', 'good', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'pass', 'normal', 'normal'])

def test_taintflag_13():
    print('Try freezing tainted flag (exception cought)', end=': ')
    reset()
    t1c.taint()
    res=R.GNAUnitTest.freeze(t1c)
    print(res and 'OK' or 'FAIL!')
    if __name__!="__main__" and  not res:
        raise Exception('freeze failed')

def test_taintflag_14():
    print('Try freezing pass through flag (exception cought)', end=': ')
    reset()
    res=R.GNAUnitTest.freeze(t2b)
    print(res and 'OK' or 'FAIL!')
    if __name__!="__main__" and  not res:
        raise Exception('freeze failed')

if __name__ == "__main__":
    for name, fcn in globals().items():
        if name.startswith('test_'):
            fcn()
