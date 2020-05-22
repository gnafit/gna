from __future__ import print_function
from tools.classwrapper import ClassWrapper
from collections import OrderedDict, Iterable

dictclasses = (dict, OrderedDict)

class DictWrapper(ClassWrapper):
    """Dictionary wrapper managing nested dictionaries
        The following functionality is implemented:
        - Tuple keys are treated to access nested dictionaries ('key1', 'key2', 'key3')
        - Optionally split symbol may be set to automatically split string keys into tuple keys:
          'key1.key2.key3' will be treated as a nested key if '.' is set for the split symbol
        - self._ may be used to access nested dictionaries via attributes: dw.key1.key2.key3
    """
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
        self._ = DictWrapperAccess(self)

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
            sub = DictWrapper(sub, parent=self)
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
            sub = DictWrapper(sub, parent=self)
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

    def walkitems(self):
        for k, v in self.items():
            k = k,
            if isinstance(v, dictclasses):
                v = DictWrapper(v, parent=self)
                for k1, v1 in v.walkitems():
                    yield k+k1, v1
            else:
                if isinstance(v, dictclasses):
                    v = DictWrapper(v, parent=self)
                yield k, v

    def walkkeys(self):
        for k, v in self.walkitems():
            yield k

    def walkvalues(self):
        for k, v in self.walkitems():
            yield v

    def visit(self, fcn):
        for k, v in self.walkitems():
            fcn(k, v)

class DictWrapperAccess(object):
    '''DictWrapper wrapper. Enables attribute based access to nested dictionaries'''
    _dict = None
    def __init__(self, dct):
        self.__dict__['_dict'] = dct

    def __getattr__(self, key):
        ret = self._dict[key]

        if isinstance(ret, DictWrapper):
            return ret._

        return ret

    def __setattr__(self, key, value):
        self._dict[key]=value

