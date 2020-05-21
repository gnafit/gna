# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import time

def timeit(fcn, n=1, dummy=None, pre=None, pre_dummy=None):
    t1 = time.clock()
    for i in xrange(n):
        if pre:
            pre()
        fcn()
    t1 = time.clock()-t1

    if dummy is None:
        dummy=fcn

    t2 = time.clock()
    for i in xrange(n):
        if pre_dummy:
            pre_dummy()
        dummy()
    t2 = time.clock()-t2

    return t1-t2

def report(fcn, n=1, *args, **kwargs):
    fmt = kwargs.pop('fmt', 'Execution time: {total} s for {count} updates, {single} s per update')
    t=timeit(fcn, n, *args, **kwargs)
    print(fmt.format(total=t, count=n, single=t/n))
    return t
