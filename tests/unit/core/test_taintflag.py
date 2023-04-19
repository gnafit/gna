#!/usr/bin/env python

import numpy as N
import pytest

from loadfast import ROOT as R

first_level_taintflag_a = R.taintflag()
first_level_taintflag_b = R.taintflag()
first_level_taintflag_c = R.taintflag()

second_level_taintflag_a = R.taintflag()
first_level_taintflag_a.subscribe(second_level_taintflag_a)
first_level_taintflag_b.subscribe(second_level_taintflag_a)

second_level_taintflag_b = R.taintflag()
first_level_taintflag_b.subscribe(second_level_taintflag_b)
first_level_taintflag_c.subscribe(second_level_taintflag_b)

third_level_taintflag_a = R.taintflag()
second_level_taintflag_a.subscribe(third_level_taintflag_a)

third_level_taintflag_b = R.taintflag()
second_level_taintflag_b.subscribe(third_level_taintflag_b)

flags = [first_level_taintflag_a, first_level_taintflag_b, first_level_taintflag_c, second_level_taintflag_a, second_level_taintflag_b, third_level_taintflag_a, third_level_taintflag_b]

def reset():
    for t in flags:
        t.set(False)

def printall(*args):
    if args:
        first_level_taintflag_a, first_level_taintflag_b, first_level_taintflag_c, second_level_taintflag_a, second_level_taintflag_b, third_level_taintflag_a, third_level_taintflag_b = args
        print("[{:^9s}]──────┐     ".format(first_level_taintflag_a))
        print("            [{:^9s}]────[{:^9s}]".format(second_level_taintflag_a, third_level_taintflag_a))
        print("[{:^9s}]──────┤     ".format(first_level_taintflag_b))
        print("            [{:^9s}]────[{:^9s}]".format(second_level_taintflag_b, third_level_taintflag_b))
        print("[{:^9s}]──────┘     ".format(first_level_taintflag_c))
    else:
        printall('first_level_taintflag_a', 'first_level_taintflag_b', 'first_level_taintflag_c', 'second_level_taintflag_a', 'second_level_taintflag_b', 'third_level_taintflag_a', 'third_level_taintflag_b')

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


@pytest.mark.serial
def test_taintflag_01():
    printboth(*flags, flags=['tainted', 'tainted', 'tainted', 'tainted', 'tainted', 'tainted', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_02():
    print('Reset')
    reset()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_03():
    print('Taint first_level_taintflag_a')
    first_level_taintflag_a.taint()
    printboth(*flags, flags=['tainted', 'good', 'good', 'tainted', 'good', 'tainted', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_04():
    print('Taint first_level_taintflag_c')
    reset()
    first_level_taintflag_c.taint()
    printboth(*flags, flags=['good', 'good', 'tainted', 'good', 'tainted', 'good', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_05():
    print('Taint first_level_taintflag_a and first_level_taintflag_c')
    reset()
    first_level_taintflag_a.taint()
    first_level_taintflag_c.taint()
    printboth(*flags, flags=['tainted', 'good', 'tainted', 'tainted', 'tainted', 'tainted', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_06():
    print('Taint first_level_taintflag_a and second_level_taintflag_a')
    reset()
    first_level_taintflag_a.taint()
    second_level_taintflag_a.taint()
    printboth(*flags, flags=['tainted', 'good', 'good', 'tainted', 'good', 'tainted', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_07():
    print('Freeze second_level_taintflag_b')
    reset()
    second_level_taintflag_b.freeze()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'frozen', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_08():
    print('Taint first_level_taintflag_c')
    reset()
    first_level_taintflag_c.taint()
    printboth(*flags, flags=['good', 'good', 'tainted', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'frozen*', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_09():
    print('Unfreeze second_level_taintflag_b')
    second_level_taintflag_b.unfreeze()
    printboth(*flags, flags=['good', 'good', 'tainted', 'good', 'tainted', 'good', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_10():
    print('Set second_level_taintflag_b pass through')
    reset()
    second_level_taintflag_b.set_pass_through()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'good', 'good', 'good'], statuses=['normal', 'normal', 'normal', 'normal', 'pass', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_11():
    print('Taint first_level_taintflag_b')
    reset()
    first_level_taintflag_b.taint()
    printboth(*flags, flags=['good', 'tainted', 'good', 'tainted', 'tainted', 'tainted', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'pass', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_12():
    print('Taint second_level_taintflag_b')
    reset()
    second_level_taintflag_b.taint()
    printboth(*flags, flags=['good', 'good', 'good', 'good', 'tainted', 'good', 'tainted'], statuses=['normal', 'normal', 'normal', 'normal', 'pass', 'normal', 'normal'])

@pytest.mark.serial
def test_taintflag_13():
    print('Try freezing tainted flag (exception cought)', end=': ')
    reset()
    first_level_taintflag_c.taint()
    res=R.GNAUnitTest.freeze(first_level_taintflag_c)
    print(res and 'OK' or 'FAIL!')
    if __name__!="__main__" and  not res:
        raise Exception('freeze failed')

@pytest.mark.serial
def test_taintflag_14():
    print('Try freezing pass through flag (exception cought)', end=': ')
    reset()
    res=R.GNAUnitTest.freeze(second_level_taintflag_b)
    print(res and 'OK' or 'FAIL!')
    if __name__!="__main__" and  not res:
        raise Exception('freeze failed')

@pytest.mark.serial
def test_taintflag_dependance_01():
    reset()

    assert third_level_taintflag_a.depends(third_level_taintflag_a)
    assert third_level_taintflag_a.depends(first_level_taintflag_a)
    assert third_level_taintflag_a.depends(first_level_taintflag_b)
    assert third_level_taintflag_a.depends(second_level_taintflag_a)
    assert not third_level_taintflag_a.depends(first_level_taintflag_c)
    assert not third_level_taintflag_a.depends(second_level_taintflag_b)
    assert not third_level_taintflag_a.depends(third_level_taintflag_b)

    assert second_level_taintflag_a.depends(first_level_taintflag_a)
    assert second_level_taintflag_a.depends(first_level_taintflag_b)
    assert second_level_taintflag_a.depends(second_level_taintflag_a)
    assert not second_level_taintflag_a.depends(first_level_taintflag_c)
    assert not second_level_taintflag_a.depends(second_level_taintflag_b)
    assert not second_level_taintflag_a.depends(third_level_taintflag_a)
    assert not second_level_taintflag_a.depends(third_level_taintflag_b)

@pytest.mark.serial
def test_taintflag_distance_01():
    reset()

    printboth(*flags)

    inf = N.uintp(-1)
    assert third_level_taintflag_a.distance(first_level_taintflag_a)==2
    assert third_level_taintflag_a.distance(first_level_taintflag_b)==2
    assert third_level_taintflag_a.distance(first_level_taintflag_c)==inf
    assert third_level_taintflag_a.distance(second_level_taintflag_a)==1
    assert third_level_taintflag_a.distance(second_level_taintflag_b)==inf
    assert third_level_taintflag_a.distance(third_level_taintflag_a)==0
    assert third_level_taintflag_a.distance(third_level_taintflag_b)==inf

    assert third_level_taintflag_b.distance(first_level_taintflag_a)==inf
    assert third_level_taintflag_b.distance(first_level_taintflag_b)==2
    assert third_level_taintflag_b.distance(first_level_taintflag_c)==2
    assert third_level_taintflag_b.distance(second_level_taintflag_a)==inf
    assert third_level_taintflag_b.distance(second_level_taintflag_b)==1
    assert third_level_taintflag_b.distance(third_level_taintflag_a)==inf
    assert third_level_taintflag_b.distance(third_level_taintflag_b)==0

    assert second_level_taintflag_a.distance(first_level_taintflag_a)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_b)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_c)==inf
    assert second_level_taintflag_a.distance(second_level_taintflag_a)==0
    assert second_level_taintflag_a.distance(second_level_taintflag_b)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_a)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_b)==inf

    assert second_level_taintflag_b.distance(first_level_taintflag_a)==inf
    assert second_level_taintflag_b.distance(first_level_taintflag_b)==1
    assert second_level_taintflag_b.distance(first_level_taintflag_c)==1
    assert second_level_taintflag_b.distance(second_level_taintflag_a)==inf
    assert second_level_taintflag_b.distance(second_level_taintflag_b)==0
    assert second_level_taintflag_b.distance(third_level_taintflag_a)==inf
    assert second_level_taintflag_b.distance(third_level_taintflag_b)==inf

@pytest.mark.serial
def test_taintflag_distance_02():
    reset()

    inf = N.uintp(-1)
    assert third_level_taintflag_a.distance(first_level_taintflag_a, True)==2
    assert third_level_taintflag_a.distance(first_level_taintflag_b, True)==2
    assert third_level_taintflag_a.distance(first_level_taintflag_c, True)==inf
    assert third_level_taintflag_a.distance(second_level_taintflag_a, True)==1
    assert third_level_taintflag_a.distance(second_level_taintflag_b, True)==inf
    assert third_level_taintflag_a.distance(third_level_taintflag_a, True)==0
    assert third_level_taintflag_a.distance(third_level_taintflag_b, True)==inf

    assert third_level_taintflag_b.distance(first_level_taintflag_a, True)==inf
    assert third_level_taintflag_b.distance(first_level_taintflag_b, True)==1   # distance throught passthrough flag
    assert third_level_taintflag_b.distance(first_level_taintflag_c, True)==1   # distance throught passthrough flag
    assert third_level_taintflag_b.distance(second_level_taintflag_a, True)==inf
    assert third_level_taintflag_b.distance(second_level_taintflag_b, True)==1   # distance to passthrough flag (not through)
    assert third_level_taintflag_b.distance(third_level_taintflag_a, True)==inf
    assert third_level_taintflag_b.distance(third_level_taintflag_b, True)==0

    assert second_level_taintflag_a.distance(first_level_taintflag_a, True)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_b, True)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_c, True)==inf
    assert second_level_taintflag_a.distance(second_level_taintflag_a, True)==0
    assert second_level_taintflag_a.distance(second_level_taintflag_b, True)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_a, True)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_b, True)==inf

    assert second_level_taintflag_b.distance(first_level_taintflag_a, True)==inf
    assert second_level_taintflag_b.distance(first_level_taintflag_b, True)==1   # distance from passthrough flag (not through)
    assert second_level_taintflag_b.distance(first_level_taintflag_c, True)==1   # distance from passthrough flag (not through)
    assert second_level_taintflag_b.distance(second_level_taintflag_a, True)==inf
    assert second_level_taintflag_b.distance(second_level_taintflag_b, True)==0
    assert second_level_taintflag_b.distance(third_level_taintflag_a, True)==inf
    assert second_level_taintflag_b.distance(third_level_taintflag_b, True)==inf

@pytest.mark.serial
def test_taintflag_distance_03():
    reset()

    inf = N.uintp(-1)
    assert third_level_taintflag_a.distance(first_level_taintflag_a,False,1)==inf
    assert third_level_taintflag_a.distance(first_level_taintflag_b,False,1)==inf
    assert third_level_taintflag_a.distance(first_level_taintflag_c,False,1)==inf
    assert third_level_taintflag_a.distance(second_level_taintflag_a,False,1)==1
    assert third_level_taintflag_a.distance(second_level_taintflag_b,False,1)==inf
    assert third_level_taintflag_a.distance(third_level_taintflag_a,False,1)==0
    assert third_level_taintflag_a.distance(third_level_taintflag_b,False,1)==inf

    assert third_level_taintflag_b.distance(first_level_taintflag_a,False,1)==inf
    assert third_level_taintflag_b.distance(first_level_taintflag_b,False,1)==inf
    assert third_level_taintflag_b.distance(first_level_taintflag_c,False,1)==inf
    assert third_level_taintflag_b.distance(second_level_taintflag_a,False,1)==inf
    assert third_level_taintflag_b.distance(second_level_taintflag_b,False,1)==1
    assert third_level_taintflag_b.distance(third_level_taintflag_a,False,1)==inf
    assert third_level_taintflag_b.distance(third_level_taintflag_b,False,1)==0

    assert second_level_taintflag_a.distance(first_level_taintflag_a,False,1)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_b,False,1)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_c,False,1)==inf
    assert second_level_taintflag_a.distance(second_level_taintflag_a,False,1)==0
    assert second_level_taintflag_a.distance(second_level_taintflag_b,False,1)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_a,False,1)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_b,False,1)==inf

    assert second_level_taintflag_b.distance(first_level_taintflag_a,False,1)==inf
    assert second_level_taintflag_b.distance(first_level_taintflag_b,False,1)==1
    assert second_level_taintflag_b.distance(first_level_taintflag_c,False,1)==1
    assert second_level_taintflag_b.distance(second_level_taintflag_a,False,1)==inf
    assert second_level_taintflag_b.distance(second_level_taintflag_b,False,1)==0
    assert second_level_taintflag_b.distance(third_level_taintflag_a,False,1)==inf
    assert second_level_taintflag_b.distance(third_level_taintflag_b,False,1)==inf


@pytest.mark.serial
def test_taintflag_distance_04():
    reset()

    inf = N.uintp(-1)
    assert third_level_taintflag_a.distance(first_level_taintflag_a, True, 1)==inf
    assert third_level_taintflag_a.distance(first_level_taintflag_b, True, 1)==inf
    assert third_level_taintflag_a.distance(first_level_taintflag_c, True, 1)==inf
    assert third_level_taintflag_a.distance(second_level_taintflag_a, True, 1)==1
    assert third_level_taintflag_a.distance(second_level_taintflag_b, True, 1)==inf
    assert third_level_taintflag_a.distance(third_level_taintflag_a, True, 1)==0
    assert third_level_taintflag_a.distance(third_level_taintflag_b, True, 1)==inf

    assert third_level_taintflag_b.distance(first_level_taintflag_a, True, 1)==inf
    assert third_level_taintflag_b.distance(first_level_taintflag_b, True, 1)==1   # distance throught passthrough flag
    assert third_level_taintflag_b.distance(first_level_taintflag_c, True, 1)==1   # distance throught passthrough flag
    assert third_level_taintflag_b.distance(second_level_taintflag_a, True, 1)==inf
    assert third_level_taintflag_b.distance(second_level_taintflag_b, True, 1)==1   # distance to passthrough flag (not through)
    assert third_level_taintflag_b.distance(third_level_taintflag_a, True, 1)==inf
    assert third_level_taintflag_b.distance(third_level_taintflag_b, True, 1)==0

    assert second_level_taintflag_a.distance(first_level_taintflag_a, True, 1)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_b, True, 1)==1
    assert second_level_taintflag_a.distance(first_level_taintflag_c, True, 1)==inf
    assert second_level_taintflag_a.distance(second_level_taintflag_a, True, 1)==0
    assert second_level_taintflag_a.distance(second_level_taintflag_b, True, 1)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_a, True, 1)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_b, True, 1)==inf

    assert second_level_taintflag_b.distance(first_level_taintflag_a, True, 1)==inf
    assert second_level_taintflag_b.distance(first_level_taintflag_b, True, 1)==1   # distance from passthrough flag (not through)
    assert second_level_taintflag_b.distance(first_level_taintflag_c, True, 1)==1   # distance from passthrough flag (not through)
    assert second_level_taintflag_b.distance(second_level_taintflag_a, True, 1)==inf
    assert second_level_taintflag_b.distance(second_level_taintflag_b, True, 1)==0
    assert second_level_taintflag_b.distance(third_level_taintflag_a, True, 1)==inf
    assert second_level_taintflag_b.distance(third_level_taintflag_b, True, 1)==inf

@pytest.mark.serial
def test_taintflag_distance_05():
    reset()

    inf = N.uintp(-1)
    assert third_level_taintflag_a.distance(first_level_taintflag_a, True, 0)==inf
    assert third_level_taintflag_a.distance(first_level_taintflag_b, True, 0)==inf
    assert third_level_taintflag_a.distance(first_level_taintflag_c, True, 0)==inf
    assert third_level_taintflag_a.distance(second_level_taintflag_a, True, 0)==inf
    assert third_level_taintflag_a.distance(second_level_taintflag_b, True, 0)==inf
    assert third_level_taintflag_a.distance(third_level_taintflag_a, True, 0)==0
    assert third_level_taintflag_a.distance(third_level_taintflag_b, True, 0)==inf

    assert third_level_taintflag_b.distance(first_level_taintflag_a, True, 0)==inf
    assert third_level_taintflag_b.distance(first_level_taintflag_b, True, 0)==inf   # distance throught passthrough flag
    assert third_level_taintflag_b.distance(first_level_taintflag_c, True, 0)==inf   # distance throught passthrough flag
    assert third_level_taintflag_b.distance(second_level_taintflag_a, True, 0)==inf
    assert third_level_taintflag_b.distance(second_level_taintflag_b, True, 0)==inf   # distance to passthrough flag (not through)
    assert third_level_taintflag_b.distance(third_level_taintflag_a, True, 0)==inf
    assert third_level_taintflag_b.distance(third_level_taintflag_b, True, 0)==0

    assert second_level_taintflag_a.distance(first_level_taintflag_a, True, 0)==inf
    assert second_level_taintflag_a.distance(first_level_taintflag_b, True, 0)==inf
    assert second_level_taintflag_a.distance(first_level_taintflag_c, True, 0)==inf
    assert second_level_taintflag_a.distance(second_level_taintflag_a, True, 0)==0
    assert second_level_taintflag_a.distance(second_level_taintflag_b, True, 0)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_a, True, 0)==inf
    assert second_level_taintflag_a.distance(third_level_taintflag_b, True, 0)==inf

    assert second_level_taintflag_b.distance(first_level_taintflag_a, True, 0)==inf
    assert second_level_taintflag_b.distance(first_level_taintflag_b, True, 0)==inf   # distance from passthrough flag (not through)
    assert second_level_taintflag_b.distance(first_level_taintflag_c, True, 0)==inf   # distance from passthrough flag (not through)
    assert second_level_taintflag_b.distance(second_level_taintflag_a, True, 0)==inf
    assert second_level_taintflag_b.distance(second_level_taintflag_b, True, 0)==0
    assert second_level_taintflag_b.distance(third_level_taintflag_a, True, 0)==inf
    assert second_level_taintflag_b.distance(third_level_taintflag_b, True, 0)==inf

# def test_taintflag_fragile():
    # t1 = R.taintflag()
    # t1.set(False)
    # fragile = R.fragile(t1)

    # print('C++ exception expected now')
    # t1.taint()
