from __future__ import print_function
from tools.classwrapper import ClassWrapper
from collections import OrderedDict, Iterable

dictclasses = (dict, OrderedDict)

class DictWrapper(ClassWrapper):
    """Dictionary wrapper managing nested dictionaries"""
    _split = None
    _parent = None
    _type = OrderedDict
    def __new__(cls, dic, *args, **kwargs):
        if not isinstance(dic, dictclasses):
            return dic
        return ClassWrapper.__new__(cls)

    def __init__(self, dic, split=None, parent=None):
        self._split = split
        if parent:
            self._parent = parent
            self._split = parent._split
        ClassWrapper.__init__(self, dic, DictWrapper)

    def parent(self):
        return self._parent

    def child(self, key):
        try:
            ret = self.get(key)
        except KeyError:
            ret = self[key]=self._type()
            return DictWrapper(ret, parent=self)

        if not isinstance(ret, DictWrapper):
            raise KeyError('Child {!s} is not DictWrapper'.format(key))

        return ret

    def iterkey(self, key):
        if isinstance(key, basestring):
            if self._split:
                for s in key.split(self._split):
                    yield s
            else:
                yield key
        elif isinstance(key, Iterable):
            for sk in key:
                for ssk in self.iterkey(sk):
                        yield ssk

    def splitkey(self, key):
        it = self.iterkey(key)
        try:
            return next(it), tuple(it)
        except StopIteration:
            return None, None

    def get(self, key, *args, **kwargs):
        if key is ():
            return self
        key, rest=self.splitkey(key)

        if not rest:
            return self.__getattr__('get')(key, *args, **kwargs)

        sub = self.__getattr__('get')(key)
        if sub is None:
            if args:
                return args[0]
            raise KeyError( "No nested key '%s'"%key )
        return sub.get(rest, *args, **kwargs)

    def __getitem__(self, key):
        if key is ():
            return self
        key, rest=self.splitkey(key)

        sub = self.__getattr__('__getitem__')(key)
        if not rest:
            return sub

        if sub is None:
            if args:
                return args[0]
            raise KeyError( "No nested key '%s'"%key )

        return sub[rest]

    def setdefault(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            self.setdefault(key, value)
            return

        if key in self:
            sub = self.__getattr__('get')(key)
        else:
            sub = self._obj[key] = self._type()
            sub = DictWrapper(sub)
            # # cfg._set_parent( self )

        sub.setdefault(rest, value)

    def set(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            self._obj[key] = value
            return

        if key in self:
            sub = self.__getattr__('get')(key)
        else:
            sub = self._obj[key] = self._type()
            sub = DictWrapper(sub)
            # # cfg._set_parent( self )

        sub.set(rest, value)

    __setitem__= set

    def __contains__(self, key):
        key, rest=self.splitkey(key)

        if not key in self._obj:
            return False

        if rest:
            sub = self.__getattr__('get')(key)
            return rest in sub

        return True


if __name__ == "__main__":
    d = dict(a=1, b=2, c=dict(d=1))
    dw = DictWrapper(d, split='.')

    import IPython; IPython.embed()
