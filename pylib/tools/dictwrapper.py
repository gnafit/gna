from __future__ import print_function
from tools.classwrapper import ClassWrapper
from collections import OrderedDict, Iterable

dictclasses = (dict, OrderedDict)

class DictWrapper(ClassWrapper):
    _split = None
    _parent = None
    def __new__(cls, dic, *args, **kwargs):
        if not isinstance(dic, dictclasses):
            return dic
        return ClassWrapper.__new__(cls)

    def __init__(self, dic, split=None):
        self._split = split
        ClassWrapper.__init__(self, dic, DictWrapper)

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
            return self._obj.get(key, *args, **kwargs)

        sub = self.__getattr__('get')(key)
        if sub is None:
            if args:
                return args[0]
            raise KeyError( "No nested key '%s'"%key )
        return sub.get( rest, *args, **kwargs )

    def set(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            self._obj[key] = value
            return

        if key in self:
            sub = self.__getattr__('get')(key)
        else:
            sub = self._obj[key] = {}
            sub = DictWrapper(sub)
            # # cfg._set_parent( self )

        sub.set(rest, value)

    __setitem__= set


if __name__ == "__main__":
    d = dict(a=1, b=2, c=dict(d=1))
    dw = DictWrapper(d, split='.')

    import IPython; IPython.embed()

    # def _set_parent(self, parent):
        # super(NestedDict, self).__setattr__('__parent__', parent)

    # def parent(self, n=1):
        # if n==1:
            # return self.__parent__

        # if n==0:
            # return self

        # if n<0:
            # raise Exception('Invalid parent depth')

        # if self.__parent__ is None:
            # raise Exception('No parent')

        # return self.__parent__.parent( n-1 )

    # def parent_key(self):
        # if self.__parent__ is None:
            # return None

        # for k, v in self.__parent__.items():
            # if v is self:
                # return k

        # raise KeyError( "Failed to determine own key in the parent dictionary" )

    # def __getitem__(self, key):
        # key, rest=process_key(key)
        # if rest:
            # return self.__storage__.__getitem__(key).__getitem__( rest )

        # if key is ():
            # return self

        # try:
            # return self.__storage__.__getitem__(key)
        # except KeyError as e:
            # if meta[self].get('createmissing', False):
                # return self(key)

            # raise

    # def setdefault(self, key, value):
        # if isinstance(value, dict):
            # value = NestedDict(value)
        # if isinstance(value, NestedDict):
            # value._set_parent( self )

        # key, rest=process_key(key)
        # if rest:
            # if not key in self.__storage__:
                # cfg = self.__storage__[key]=NestedDict()
                # cfg._set_parent( self )
                # return cfg.setdefault( rest, value )
            # return self.__storage__.get(key).setdefault( rest, value )

        # return self.__storage__.setdefault(key, value)

    # def values(self, nested=False):
        # for v in self.__storage__.values():
            # if nested and isinstance(v, NestedDict):
                # for nv in v.values(nested=True):
                    # yield nv
            # else:
                # yield v

    # def items(self, nested=False):
        # if nested:
            # for k, v in self.__storage__.items():
                # if isinstance(v, NestedDict):
                    # for nk, nv in v.items(nested=True):
                        # yield (k,)+nk, nv
                # else:
                    # yield (k,), v
        # else:
            # for k, v in self.__storage__.items():
                # yield k, v

    # def keys(self, nested=False):
        # if nested:
            # for k, v in self.__storage__.items():
                # if isinstance(v, NestedDict):
                    # for nk in v.keys(nested=True):
                        # yield (k,)+nk
                # else:
                    # yield k,
        # else:
            # for k in self.__storage__.keys():
                # yield k

    # def __contains__(self, key):
        # key, rest=process_key(key)

        # if not self.__storage__.__contains__(key):
            # return False

        # if rest:
            # return self.__storage__.get(key).__contains__(rest)

        # return True

    # def __call__(self, key):
        # if isinstance( key, (list, tuple) ):
            # key, rest = key[0], key[1:]
            # if rest:
                # if not key in self.__storage__:
                    # cfg = self.__storage__[key]=NestedDict()
                    # cfg._set_parent( self )
                    # return cfg.__call__( rest )
                # return self.__storage__.get(key).__call__(rest)

        # if isinstance( key, basestring ):
            # if '.' in key:
                # return self.__call__(key.split('.'))

        # other = self.__storage__.get(key, None)
        # if other is None:
            # value = self.__storage__[key] = NestedDict()
            # value._set_parent( self )
            # return value

        # if isinstance(other, NestedDict):
            # return other

        # raise KeyError( "Can not create nested configuration as soon as soon as the key '%s' already exists"%key )


    # def __load__(self, filename, subst=[]):
        # if subst:
            # if subst=='default':
                # subst = dict( key='location', values=['config', 'config_local'] )

            # if type(subst) in [ list, tuple ]:
                # filenames = [ filename.format( s ) for s in subst ]
            # elif type(subst) is dict:
                # filenames = [ filename.format( **{ subst['key']: v } ) for v in subst['values'] ]
            # else:
                # raise Exception( "Unsupported 'subst' type "+type(subst).__name__.__repr__() )
        # else:
            # filenames = [ filename ]

        # unimportant = False
        # for filename in filenames:
            # if unimportant and not path.isfile( filename ):
                # if meta[self].get('verbose', False):
                    # print( 'Skipping nonexistent file', filename )
                # continue

            # dic = self.__load_dic__(filename, dictonly=True)
            # self.__import__(dic)
            # unimportant = True

    # def __load_dic__(self, filename, dictonly=False):
        # print('Loading config file:', filename)
        # dic =  runpy.run_path(filename, init_globals )
        # for k in init_globals:
            # if dic[k]==init_globals[k]:
                # del dic[k]

        # if dictonly:
            # return dic
        # return NestedDict(dic)

    # def __import__(self, dic):
        # for k, v in dic.items():
            # if isinstance(k, basestring) and k.startswith('__'):
                # continue
            # if meta[self].get('verbose', False):
                # if k in self:
                    # print( 'Reset', k, 'to', v.__repr__() )
                # else:
                    # print( 'Set', k, 'to', v.__repr__() )
            # self.__setattr__(k, v)
