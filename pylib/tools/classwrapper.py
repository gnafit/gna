# -*- coding: utf-8 -*-

import inspect

class ClassWrapper(object):
    """Wrap any class and forward
        - getitem/setitem
        - getattr
        - call
        calls, wrap the result if the class is of the same type"""
    def __init__(self, obj, parent=None, types=None):
        self._obj = obj
        if types:
            self._types = types
        else:
            self._types = type(obj)
        self._wrapper_class = type(self)

    def unwrap(self):
        return self._obj

    def __str__(self):
        return str(self._obj)

    def __repr__(self):
        return repr(self._obj)

    def __dir__(self):
        return dir(self._obj)

    def __len__(self):
        return len(self._obj)

    def __bool__(self):
        return bool(self._obj)

    def __iter__(self):
        return iter(self._obj)

    def __contains__(self, v):
        return v in self._obj

    def __getattr__(self, attr):
        method = getattr(self._obj, attr)
        return self._wrap(method)

    def __getitem__(self, k):
        return self._wrap(self._obj[k])

    def __call__(self, *args, **kwargs):
        return self._wrap(self._obj(*args, **kwargs))

    def __setitem__(self, k, v):
        self._obj[k]=v

    def _wrap(self, obj):
        if isinstance(obj, ClassWrapper):
            return obj
        if inspect.isgenerator(obj) or inspect.isgeneratorfunction(obj):
            return self._wrapgenerator(obj)
        if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
            return self._wrapmethod(obj)

        return self._wrapobject(obj)

    def _wrapobject(self, obj):
        if isinstance(obj, self._types):
            return self._wrapper_class(obj, parent=self)

        return obj

    def _wrapmethod(self, method):
        def wrapped_method(*args, **kwargs):
            return self._wrap(method(*args, **kwargs))
        return wrapped_method

    def _wrapgenerator(self, generator):
        def wrapped(*args, **kwargs):
            for i in generator(*args, **kwargs):
                yield self._wrap(i)

        return wrapped
