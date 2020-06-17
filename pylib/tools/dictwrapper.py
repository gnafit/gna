# -*- coding: utf-8 -*-
from __future__ import print_function
from tools.classwrapper import ClassWrapper
from collections import OrderedDict, Iterable, MutableMapping
import inspect

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
        if not isinstance(dic, (MutableMapping, DictWrapper)):
            return dic
        return ClassWrapper.__new__(cls)

    def __init__(self, dic, split=None, parent=None, *args, **kwargs):
        if isinstance(dic, DictWrapper):
            dic = dic._obj
        self._split = split
        self._type = type(dic)
        ClassWrapper.__init__(self, dic, types=MutableMapping)
        if parent:
            self._parent = parent
            self._split = parent._split
            self._types = parent._types

    @property
    def _(self):
        return DictWrapperAccess(self)

    def parent(self):
        return self._parent

    def child(self, key):
        try:
            ret = self[key]
        except KeyError:
            ret = self[key]=self._type()
            return self._wrapobject(ret)

        if not isinstance(ret, self._wrapper_class):
            raise KeyError('Child {!s} is not DictWrapper'.format(key))

        return ret

    def keys(self):
        return self._obj.keys()

    def iterkey(self, key):
        if isinstance(key, str):
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
            raise KeyError( "No nested key '%s'"%key )

        return sub[rest]

    def __delitem__(self, key):
        if key is ():
            raise Exception('May not delete itself')
        key, rest=self.splitkey(key)

        sub = self.__getattr__('__getitem__')(key)
        if not rest:
            del self._obj[key]
            return

        del sub[rest]

    def setdefault(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            self.setdefault(key, value)
            return

        if key in self:
            sub = self.__getattr__('get')(key)
        else:
            sub = self._obj[key] = self._type()
            sub = self._wrapobject(sub)
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
            sub = self._wrapobject(sub)
            # # cfg._set_parent( self )

        sub.set(rest, value)

    __setitem__= set

    def __contains__(self, key):
        if key is ():
            return True
        key, rest=self.splitkey(key)

        if not key in self._obj:
            return False

        if rest:
            sub = self.__getattr__('get')(key)
            return rest in sub

        return True

    def items(self):
        for k, v in self._obj.items():
            yield k, self._wrapobject(v)

    def deepcopy(self):
        new = DictWrapper(self._type())
        for k, v in self.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                new[k] = v.deepcopy()._obj
            else:
                new[k] = v

        new._split = self._split

        return new

    def walkitems(self, startfromkey=(), appendstartkey=False):
        v0 = self[startfromkey]
        k0 = tuple(self.iterkey(startfromkey))
        if not isinstance(v0, self._wrapper_class):
            yield k0, v0
            return

        if not appendstartkey:
            k0 = ()

        for k, v in v0.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                for k1, v1 in v.walkitems():
                    yield k0+k+k1, v1
            else:
                yield k0+k, v

    def walkdicts(self):
        yieldself= True
        for k, v in self.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                yieldself=False
                for k1, v1 in v.walkdicts():
                    yield k+k1, v1
        if yieldself:
            yield (), self

    def walkkeys(self, startfromkey=()):
        for k, v in self.walkitems(startfromkey):
            yield k

    def walkvalues(self, startfromkey=()):
        for k, v in self.walkitems(startfromkey):
            yield v

    def visit(self, visitor, parentkey=()):
        visitor = MakeDictWrapperVisitor(visitor)

        if not parentkey:
            visitor.start(self)

        visitor.enterdict(parentkey, self)
        for k, v in self.items():
            key = parentkey + (k,)
            if isinstance(v, self._wrapper_class):
                v.visit(visitor, parentkey=key)
            else:
                visitor.visit(key, v)

        visitor.exitdict(parentkey, self)

        if not parentkey:
            visitor.stop(self)

class DictWrapperAccess(object):
    '''DictWrapper wrapper. Enables attribute based access to nested dictionaries'''
    _ = None
    def __init__(self, dct):
        self.__dict__['_'] = dct

    def __call__(self, key):
        return self._.child(key)._

    def __getattr__(self, key):
        ret = self._[key]

        if isinstance(ret, self._._wrapper_class):
            return ret._

        return ret

    def __setattr__(self, key, value):
        self._[key]=value

    def __delattr__(self, key):
        del self._[key]

    def __dir__(self):
        return list(self._.keys())

def MakeDictWrapperVisitor(fcn):
    if isinstance(fcn, DictWrapperVisitor):
        return fcn

    if not inspect.isfunction(fcn) and not hasattr(fcn, '__call__'):
        raise Exception('Expect function, got '+type(fcn).__name__)

    ret=DictWrapperVisitor()
    ret.visit = fcn
    return ret

class DictWrapperVisitor(object):
    def start(self, dct):
        pass

    def enterdict(self, k, v):
        pass

    def visit(self, k, v):
        pass

    def exitdict(self, k, v):
        pass

    def stop(self, dct):
        pass

class DictWrapperVisitorDemostrator(DictWrapperVisitor):
    fmt = '{action:7s} {depth!s:>5s} {key!s:<{keylen}s} {vtype!s:<{typelen}s} {value}'
    opts = dict(keylen=20, typelen=15)
    def typestring(self, v):
        return type(v).__name__

    def start(self, d):
        v = object.__repr__(d.unwrap())
        print('Start printing dictionary:', v)
        print(self.fmt.format(action='Action', depth='Depth', key='Key', vtype='Type', value='Value', **self.opts))

    def stop(self, d):
        print('Done printing dictionary')

    def enterdict(self, k, d):
        d = d.unwrap()
        v = object.__repr__(d)
        print(self.fmt.format(action='Enter', depth=len(k), key=k, vtype=self.typestring(d), value=v, **self.opts))

    def exitdict(self, k, d):
        d = d.unwrap()
        v = object.__repr__(d)
        print(self.fmt.format(action='Exit', depth=len(k), key=k, vtype=self.typestring(d), value=v, **self.opts))

    def visit(self, k, v):
        print(self.fmt.format(action='Visit', depth=len(k), key=k, vtype=self.typestring(v), value=v, **self.opts))
