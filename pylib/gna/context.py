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

def current_precision_allocator():
    return R.arrayviewAllocator(_current_precision)

def current_precision_manager():
    return R.GNA.TreeManager(_current_precision)

class precision(object):
    """Context manager for the floating precision"""
    old_precision=''
    def __init__(self, precision):
        self.precision=precision

    def __enter__(self):
        if self.precision!=_current_precision:
            if not current_precision_allocator():
                raise Exception('may not change precision while allocator is set')

        self.old_precision = _current_precision
        _set_current_precision(self.precision)

    def __exit__(self, *args):
        _set_current_precision(self.old_precision)

class cuda(object):
    """Context manager for GPU
    Makes Initializer to switch transformations to "gpu" function after initialization"""
    backup_function=''
    def __init__(self, enabled=True):
        self.handle=R.TransformationTypes.InitializerBase
        self._enabled = enabled

    def __enter__(self):
        if not self._enabled:
            return

        self.backup_function = self.handle.getDefaultFunction()
        self.handle.setDefaultFunction('gpu')

    def __exit__(self, *args):
        if not self._enabled:
            return

        self.handle.setDefaultFunction(self.backup_function)

class allocator(object):
    """Set allocator for arrayview"""
    backup_allocator=None
    def __init__(self, allocator):
        if isinstance(allocator, int):
            self.allocator = R.arrayviewAllocatorSimple(_current_precision)(allocator)
        else:
            self.allocator = allocator

    def __enter__(self):
        self.cls = current_precision_allocator()
        self.backup_allocator = self.cls.current()
        self.cls.setCurrent(self.allocator)

        return self.allocator

    def __exit__(self, *args):
        self.cls.setCurrent(self.backup_allocator)

class manager(object):
    """Set TreeManager"""
    def __init__(self, manager=None):
        cls = current_precision_manager()
        if isinstance(manager, (int, None)):
            self.manager = cls(manager)
        else:
            self.manager = manager or cls()

    def __enter__(self):
        self.manager.makeCurrent()
        return self.manager

    def __exit__(self, *args):
        self.manager.resetCurrent()

class set_context(object):
    """Set multiple contexts using a single one"""
    def __init__(self, **kwargs):
        self.gpu=kwargs.pop('gpu', None)
        self.manager=kwargs.pop('manager', None)
        self.precision=kwargs.pop('precision', None)
        assert not kwargs, 'Unknown options: '+str(kwargs)

        self.chain = []
        self.rets = []

        if self.precision:
            assert self.precision in ('float', 'double'), 'Unsupported precision: '+self.precision
            self.chain.append(precision(self.precision))

        if self.manager:
            assert isinstance(self.manager, int)
            self.chain.append(manager(self.manager))

        if self.gpu:
            assert self.manager and self.gpu, 'For GPU support manager should be also set'
            self.chain.append(cuda(enabled=self.gpu))

    def __enter__(self):
        for cntx in self.chain:
            rets = cntx.__enter__()
            if rets:
                self.rets.append(rets)

        lenrets = len(self.rets)
        if lenrets==1:
            return self.rets[0]
        elif lenrets==0:
            return

        return tuple(self.rets)

    def __exit__(self, *args, **kwargs):
        for cntx in reversed(self.chain):
            cntx.__exit__(*args, **kwargs)

