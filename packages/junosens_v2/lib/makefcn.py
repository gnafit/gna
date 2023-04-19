import numpy as np

def MakeFcn(arg, ret):
    """Make a python function based on outputs"""
    dt1=arg.datatype()
    dt2=ret.datatype()
    assert dt1==dt2

    def fcn(arrn):
        resultn=np.zeros_like(arrn)
        result=resultn.ravel()
        arr = np.ascontiguousarray(arrn, dtype='d')

        frozen1=arg.getTaintflag().frozen()
        assert not frozen1, "May not work with frozen outputs"

        narg=dt1.size()
        nresult=arr.size

        step=min(narg, nresult)
        ptr=0
        while ptr<nresult:
            end=min(ptr+step, arr.size)
            arg.fill(arr[ptr:end])

            slc=ret.data().ravel(order='F')
            result[ptr:end]=slc[:step]
            # print(f'0:{step} -> {ptr}:{end}')

            ptr=end

        arg.unfreeze()

        return resultn

    return fcn

def MakeFcn2(arg1, arg2, ret):
    """Make a python function of two arguments based on outputs"""
    dt1=arg1.datatype()
    dt2=arg2.datatype()
    dt3=ret.datatype()
    assert dt1==dt2 and dt2==dt3

    def fcn(arr1n, arr2n):
        assert arr1n.shape==arr2n.shape
        arr1, arr2 = np.ascontiguousarray(arr1n.ravel(), dtype='d'), np.ascontiguousarray(arr2n.ravel(), dtype='d')

        frozen1=arg1.getTaintflag().frozen()
        frozen2=arg2.getTaintflag().frozen()
        assert not frozen1 and not frozen2, "May not work with frozen outputs"

        resultn=np.zeros_like(arr1n)
        result = resultn.ravel()

        narg=dt1.size()
        nresult=arr1.size

        step=min(narg, nresult)
        ptr=0
        while ptr<nresult:
            end=min(ptr+step, arr1.size)
            arg1.fill(arr1[ptr:end])
            arg2.fill(arr2[ptr:end])

            slc=ret.data().ravel(order='F')
            result[ptr:end]=slc[:step]
            # print(f'0:{step} -> {ptr}:{end}')

            ptr=end

        arg1.unfreeze()
        arg2.unfreeze()

        return resultn

    return fcn

