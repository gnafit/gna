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

    def items(self):
        for k, v in self._obj.iteritems():
            if isinstance(v, dictclasses):
                v = DictWrapper(v, parent=self)
            yield k, v
    iteritems=items

    def walkitems(self):
        for k, v in self.items():
            k = k,
            if isinstance(v, DictWrapper):
                for k1, v1 in v.walkitems():
                    yield k+k1, v1
            else:
                yield k, v

    def walkkeys(self):
        for k, v in self.walkitems():
            yield k

    def walkvalues(self):
        for k, v in self.walkitems():
            yield v

    def visit(self, visitor, parentkey=()):
        visitor = DictWrapperVisitor(visitor)

        if not parentkey:
            visitor.start(self)

        visitor.enterdict(parentkey, self)
        for k, v in self.items():
            key = parentkey + (k,)
            if isinstance(v, DictWrapper):
                v.visit(visitor, parentkey=key)
            else:
                visitor.visit(key, v)

        visitor.exitdict(parentkey, self)

        if not parentkey:
            visitor.stop(self)

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

class DictWrapperVisitor(object):
    def __new__(cls, fcn_or_visitor=None):
        if isinstance(fcn_or_visitor, DictWrapperVisitor):
            return fcn_or_visitor

        ret=object.__new__(cls)
        if fcn_or_visitor is not None:
            ret.visit = fcn_or_visitor
        return ret

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

class DictWrapperPrinter(DictWrapperVisitor):
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
        print(self.fmt.format(action='Visit', depth=len(k), key=k, vtype=self.typestring(d), value=v, **self.opts))

    def visit(self, k, v):
        print(self.fmt.format(action='Exit', depth=len(k), key=k, vtype=self.typestring(v), value=v, **self.opts))
