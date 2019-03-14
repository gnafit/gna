# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R

_current_precision = 'double'
_current_precision_short = 'double'
def _set_current_precision(precision):
    global _current_precision, _current_precision_short
    assert precision in R.GNA.provided_precisions(), 'Unsupported precision '+precision
    _current_precision=precision
    _current_precision_short=precision[0]

def current_precision():
    return _current_precision

def current_precision_short():
    return _current_precision_short

class precision(object):
    """Context manager for the floating precision"""
    old_precision=''
    def __init__(self, precision):
        self.precision=precision

    def __enter__(self):
        self.old_precision = _current_precision
        _set_current_precision(self.precision)

    def __exit__(self, *args):
        _set_current_precision(self.old_precision)

class cuda(object):
    """Context manager for GPU
    Makes Initializer to switch transformations to "gpu" function after initialization"""
    backup_function=''
    def __init__(self):
        self.handle=R.TransformationTypes.InitializerBase

    def __enter__(self):
        self.backup_function = self.handle.getDefaultFunction()
        self.handle.setDefaultFunction('gpu')

    def __exit__(self, *args):
        self.handle.setDefaultFunction(self.backup_function)

class allocator(object):
    """Set allocator for arrayview"""
    backup_function=''
    def __init__(self, allocator):
        self.allocator = allocator

    def __enter__(self):
        self.cls = R.arrayviewAllocator(_current_precision)
        self.backup_allocator = self.cls.current()
        self.cls.setCurrent(self.allocator)

    def __exit__(self, *args):
        self.cls.setCurrent(self.backup_allocator)
