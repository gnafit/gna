
import time

def timeit(fcn, n=1, dummy=None, pre=None, pre_dummy=None):
    t1 = time.perf_counter()
    for i in range(n):
        if pre:
            pre()
        fcn()
    t1 = time.perf_counter()-t1

    if dummy is None:
        dummy=fcn

    t2 = time.perf_counter()
    for i in range(n):
        if pre_dummy:
            pre_dummy()
        dummy()
    t2 = time.perf_counter()-t2

    return t1-t2

def report(fcn, n=1, *args, **kwargs):
    fmt = kwargs.pop('fmt', 'Execution time: {total} s for {count} updates, {single} s per update')
    t=timeit(fcn, n, *args, **kwargs)
    print(fmt.format(total=t, count=n, single=t/n))
    return t
