#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from loadfast import ROOT as R

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

def test_taintflag_dependance_01():
    reset()

    assert t3a.depends(t3a)
    assert t3a.depends(t1a)
    assert t3a.depends(t1b)
    assert t3a.depends(t2a)
    assert not t3a.depends(t1c)
    assert not t3a.depends(t2b)
    assert not t3a.depends(t3b)

    assert t2a.depends(t1a)
    assert t2a.depends(t1b)
    assert t2a.depends(t2a)
    assert not t2a.depends(t1c)
    assert not t2a.depends(t2b)
    assert not t2a.depends(t3a)
    assert not t2a.depends(t3b)

def test_taintflag_distance_01():
    reset()

    printboth(*flags)

    inf = N.uintp(-1)
    assert t3a.distance(t1a)==2
    assert t3a.distance(t1b)==2
    assert t3a.distance(t1c)==inf
    assert t3a.distance(t2a)==1
    assert t3a.distance(t2b)==inf
    assert t3a.distance(t3a)==0
    assert t3a.distance(t3b)==inf

    assert t3b.distance(t1a)==inf
    assert t3b.distance(t1b)==2
    assert t3b.distance(t1c)==2
    assert t3b.distance(t2a)==inf
    assert t3b.distance(t2b)==1
    assert t3b.distance(t3a)==inf
    assert t3b.distance(t3b)==0

    assert t2a.distance(t1a)==1
    assert t2a.distance(t1b)==1
    assert t2a.distance(t1c)==inf
    assert t2a.distance(t2a)==0
    assert t2a.distance(t2b)==inf
    assert t2a.distance(t3a)==inf
    assert t2a.distance(t3b)==inf

    assert t2b.distance(t1a)==inf
    assert t2b.distance(t1b)==1
    assert t2b.distance(t1c)==1
    assert t2b.distance(t2a)==inf
    assert t2b.distance(t2b)==0
    assert t2b.distance(t3a)==inf
    assert t2b.distance(t3b)==inf

def test_taintflag_distance_02():
    reset()

    inf = N.uintp(-1)
    assert t3a.distance(t1a, True)==2
    assert t3a.distance(t1b, True)==2
    assert t3a.distance(t1c, True)==inf
    assert t3a.distance(t2a, True)==1
    assert t3a.distance(t2b, True)==inf
    assert t3a.distance(t3a, True)==0
    assert t3a.distance(t3b, True)==inf

    assert t3b.distance(t1a, True)==inf
    assert t3b.distance(t1b, True)==1   # distance throught passthrough flag
    assert t3b.distance(t1c, True)==1   # distance throught passthrough flag
    assert t3b.distance(t2a, True)==inf
    assert t3b.distance(t2b, True)==1   # distance to passthrough flag (not through)
    assert t3b.distance(t3a, True)==inf
    assert t3b.distance(t3b, True)==0

    assert t2a.distance(t1a, True)==1
    assert t2a.distance(t1b, True)==1
    assert t2a.distance(t1c, True)==inf
    assert t2a.distance(t2a, True)==0
    assert t2a.distance(t2b, True)==inf
    assert t2a.distance(t3a, True)==inf
    assert t2a.distance(t3b, True)==inf

    assert t2b.distance(t1a, True)==inf
    assert t2b.distance(t1b, True)==1   # distance from passthrough flag (not through)
    assert t2b.distance(t1c, True)==1   # distance from passthrough flag (not through)
    assert t2b.distance(t2a, True)==inf
    assert t2b.distance(t2b, True)==0
    assert t2b.distance(t3a, True)==inf
    assert t2b.distance(t3b, True)==inf

def test_taintflag_distance_03():
    reset()

    inf = N.uintp(-1)
    assert t3a.distance(t1a,False,1)==inf
    assert t3a.distance(t1b,False,1)==inf
    assert t3a.distance(t1c,False,1)==inf
    assert t3a.distance(t2a,False,1)==1
    assert t3a.distance(t2b,False,1)==inf
    assert t3a.distance(t3a,False,1)==0
    assert t3a.distance(t3b,False,1)==inf

    assert t3b.distance(t1a,False,1)==inf
    assert t3b.distance(t1b,False,1)==inf
    assert t3b.distance(t1c,False,1)==inf
    assert t3b.distance(t2a,False,1)==inf
    assert t3b.distance(t2b,False,1)==1
    assert t3b.distance(t3a,False,1)==inf
    assert t3b.distance(t3b,False,1)==0

    assert t2a.distance(t1a,False,1)==1
    assert t2a.distance(t1b,False,1)==1
    assert t2a.distance(t1c,False,1)==inf
    assert t2a.distance(t2a,False,1)==0
    assert t2a.distance(t2b,False,1)==inf
    assert t2a.distance(t3a,False,1)==inf
    assert t2a.distance(t3b,False,1)==inf

    assert t2b.distance(t1a,False,1)==inf
    assert t2b.distance(t1b,False,1)==1
    assert t2b.distance(t1c,False,1)==1
    assert t2b.distance(t2a,False,1)==inf
    assert t2b.distance(t2b,False,1)==0
    assert t2b.distance(t3a,False,1)==inf
    assert t2b.distance(t3b,False,1)==inf


def test_taintflag_distance_04():
    reset()

    inf = N.uintp(-1)
    assert t3a.distance(t1a, True, 1)==inf
    assert t3a.distance(t1b, True, 1)==inf
    assert t3a.distance(t1c, True, 1)==inf
    assert t3a.distance(t2a, True, 1)==1
    assert t3a.distance(t2b, True, 1)==inf
    assert t3a.distance(t3a, True, 1)==0
    assert t3a.distance(t3b, True, 1)==inf

    assert t3b.distance(t1a, True, 1)==inf
    assert t3b.distance(t1b, True, 1)==1   # distance throught passthrough flag
    assert t3b.distance(t1c, True, 1)==1   # distance throught passthrough flag
    assert t3b.distance(t2a, True, 1)==inf
    assert t3b.distance(t2b, True, 1)==1   # distance to passthrough flag (not through)
    assert t3b.distance(t3a, True, 1)==inf
    assert t3b.distance(t3b, True, 1)==0

    assert t2a.distance(t1a, True, 1)==1
    assert t2a.distance(t1b, True, 1)==1
    assert t2a.distance(t1c, True, 1)==inf
    assert t2a.distance(t2a, True, 1)==0
    assert t2a.distance(t2b, True, 1)==inf
    assert t2a.distance(t3a, True, 1)==inf
    assert t2a.distance(t3b, True, 1)==inf

    assert t2b.distance(t1a, True, 1)==inf
    assert t2b.distance(t1b, True, 1)==1   # distance from passthrough flag (not through)
    assert t2b.distance(t1c, True, 1)==1   # distance from passthrough flag (not through)
    assert t2b.distance(t2a, True, 1)==inf
    assert t2b.distance(t2b, True, 1)==0
    assert t2b.distance(t3a, True, 1)==inf
    assert t2b.distance(t3b, True, 1)==inf

def test_taintflag_distance_05():
    reset()

    inf = N.uintp(-1)
    assert t3a.distance(t1a, True, 0)==inf
    assert t3a.distance(t1b, True, 0)==inf
    assert t3a.distance(t1c, True, 0)==inf
    assert t3a.distance(t2a, True, 0)==inf
    assert t3a.distance(t2b, True, 0)==inf
    assert t3a.distance(t3a, True, 0)==0
    assert t3a.distance(t3b, True, 0)==inf

    assert t3b.distance(t1a, True, 0)==inf
    assert t3b.distance(t1b, True, 0)==inf   # distance throught passthrough flag
    assert t3b.distance(t1c, True, 0)==inf   # distance throught passthrough flag
    assert t3b.distance(t2a, True, 0)==inf
    assert t3b.distance(t2b, True, 0)==inf   # distance to passthrough flag (not through)
    assert t3b.distance(t3a, True, 0)==inf
    assert t3b.distance(t3b, True, 0)==0

    assert t2a.distance(t1a, True, 0)==inf
    assert t2a.distance(t1b, True, 0)==inf
    assert t2a.distance(t1c, True, 0)==inf
    assert t2a.distance(t2a, True, 0)==0
    assert t2a.distance(t2b, True, 0)==inf
    assert t2a.distance(t3a, True, 0)==inf
    assert t2a.distance(t3b, True, 0)==inf

    assert t2b.distance(t1a, True, 0)==inf
    assert t2b.distance(t1b, True, 0)==inf   # distance from passthrough flag (not through)
    assert t2b.distance(t1c, True, 0)==inf   # distance from passthrough flag (not through)
    assert t2b.distance(t2a, True, 0)==inf
    assert t2b.distance(t2b, True, 0)==0
    assert t2b.distance(t3a, True, 0)==inf
    assert t2b.distance(t3b, True, 0)==inf

# def test_taintflag_fragile():
    # t1 = R.taintflag()
    # t1.set(False)
    # fragile = R.fragile(t1)

    # print('C++ exception expected now')
    # t1.taint()

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

