from schema import Schema, Or, Optional, Use, And
import os

def isreadable(filename: str):
    """Returns True if the file is readable"""
    return os.access(filename, os.R_OK)

def isrootfile(filename: str):
    """Returns True if the file extension is .root"""
    return filename.endswith('.root')

def isfilewithext(ext: str):
    """Returns a function that retunts True if the file extension is consistent"""
    def checkfilename(filename: str):
        return filename.endswith(f'.{ext}')
    return checkfilename

def haslength(*, exactly: int=None, min: int=None, max: int=None):
    """Returns a function that checks length: min<=l<=max (inclusive)"""
    def checklength(l):
        ln = len(l)
        if exactly is not None and ln!=exactly:
            return False
        if min is not None and ln<min:
            return False
        if max is not None and ln>max:
            return False
        return True

    return checklength

def isascendingarray(arr):
    """Returns true if the array is ascending"""
    return all(a<b for a,b in zip(arr[:-1], arr[1:]))

def isnonnegative(v):
    """Returns True if the value is not negative """
    return v>=0

StrOrListOfStrings = Or(list, And(str, Use(lambda s: [s])))
