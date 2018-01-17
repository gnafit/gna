"""configurator class allows to load any python file by its filename
and store the contents in a namespace
namespace elements are accessible throught both key access or member acess"""

from __future__ import print_function
import runpy
from os import path
from collections import OrderedDict
from weakref import WeakKeyDictionary
import numpy

meta = WeakKeyDictionary()
init_globals = dict( percent=0.01, numpy=numpy )

class NestedDict(object):
    __parent__ = None
    def __init__(self, iterable=None, **kwargs):
        super(NestedDict, self).__setattr__('__storage__', OrderedDict())

        meta[self] = dict()

        if iterable:
            if type(iterable) is dict:
                iterable = sorted(iterable.items())
            self.__import__(OrderedDict(iterable))

        if kwargs:
            self.__import__(OrderedDict(**kwargs))

    def __repr__(self):
        return 'NestedDict'+self.__storage__.__repr__()[11:]

    def __str__(self):
        pprint(self)
        return '' #TODO

    def __bool__(self):
        return bool(self.keys())

    def __len__(self):
        return len(self.keys())

    def _set_parent(self, parent):
        super(NestedDict, self).__setattr__('__parent__', parent)

    def parent(self):
        return self.__parent__

    def get(self, key, default=None):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                sub = self.__storage__.get(key)
                if sub is None:
                    raise KeyError( "No nested key '%s'"%key )
                return sub.get( rest, default )

        if isinstance( key, basestring ) and '.' in key:
            return self.get(key.split('.'))

        return self.__storage__.get(key, default)

    def __getitem__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__storage__.__getitem__(key).__getitem__( rest )

        if isinstance( key, basestring ) and '.' in key:
            return self.__getitem__(key.split('.'))

        return self.__storage__.__getitem__(key)

    __getattr__ = __getitem__

    def set(self, key, value, loading_from_file=False ):
        if isinstance(value, dict):
            value = NestedDict(value)
        if isinstance(value, NestedDict):
            value._set_parent( self )

        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__storage__:
                    cfg = self.__storage__[key]=NestedDict()
                    cfg._set_parent( self )
                    return cfg.set( rest, value )
                return self.__storage__.get(key).set( rest, value )

        if isinstance( key, basestring ) and '.' in key:
            return self.set( key.split('.'), value )

        self.__storage__[key] = value

    __setattr__ = set
    __setitem__= set

    def setdefault(self, key, value):
        if isinstance(value, dict):
            value = NestedDict(value)
        if isinstance(value, NestedDict):
            value._set_parent( self )

        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__storage__:
                    cfg = self.__storage__[key]=NestedDict()
                    cfg._set_parent( self )
                    return cfg.setdefault( rest, value )
                return self.__storage__.get(key).setdefault( rest, value )

        if isinstance( key, basestring ):
            if '.' in key:
                return self.setdefault(key.split('.'), value)

        return self.__storage__.setdefault(key, value)

    def keys(self):
        return self.__storage__.keys()

    def values(self):
        return self.__storage__.values()

    def items(self):
        return self.__storage__.items()

    def __contains__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__storage__.get(key).__contains__(rest)

        if isinstance( key, basestring ):
            if '.' in key:
                return self.__contains__(key.split('.'))

        return self.__storage__.__contains__(key)

    def __call__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__storage__:
                    cfg = self.__storage__[key]=NestedDict()
                    cfg._set_parent( self )
                    return cfg.__call__( rest )
                return self.__storage__.get(key).__call__(rest)

        if isinstance( key, basestring ):
            if '.' in key:
                return self.__call__(key.split('.'))

        if self.__storage__.__contains__( key ):
            raise KeyError( "Can not create nested configuration as the key '%s' already exists"%key )

        value = self.__storage__[key] = NestedDict()
        value._set_parent( self )
        return value

    def __load__(self, filename, subst=[]):
        if subst:
            if subst=='default':
                subst = dict( key='location', values=['config', 'config_local'] )

            if type(subst) in [ list, tuple ]:
                filenames = [ filename.format( s ) for s in subst ]
            elif type(subst) is dict:
                filenames = [ filename.format( **{ subst['key']: v } ) for v in subst['values'] ]
            else:
                raise Exception( "Unsupported 'subst' type "+type(subst).__name__.__repr__() )
        else:
            filenames = [ filename ]

        unimportant = False
        for filename in filenames:
            if unimportant and not path.isfile( filename ):
                if meta[self].get('verbose', False):
                    print( 'Skipping nonexistent file', filename )
                continue
            dic = self.__load_dic__(filename, dictonly=True)
            self.__import__(dic)
            unimportant = True

    def __load_dic__(self, filename, dictonly=False):
        print('Loading config file:', filename)
        dic =  runpy.run_path(filename, init_globals )
        for k in init_globals:
            if dic[k]==init_globals[k]:
                del dic[k]

        if dictonly:
            return dic
        return NestedDict(dic)

    def __import__(self, dic):
        for k, v in dic.items():
            if isinstance(k, str) and k.startswith('__'):
                continue
            if meta[self].get('verbose', False):
                if k in self:
                    print( 'Reset', k, 'to', v.__repr__() )
                else:
                    print( 'Set', k, 'to', v.__repr__() )
            self.__setattr__(k, v)

def configurator(filename=None, dic={}, **kwargs):
    self = NestedDict()

    if filename:
        self['@loaded_from']=filename

    meta[self]['verbose']=kwargs.pop( 'debug', False )
    if filename:
        self.__load__(filename, **kwargs)
    elif dic:
        self.__import__(dic)

    return self

def pprint( nd, margin='', nested=False ):
    if nd:
        print('${')
        margin+='  '
        for k, v in nd.items():
            print( margin, k, ' : ', end='', sep='' )
            if isinstance( v, NestedDict ):
                pprint( v, margin, nested=True )
            else:
                print( str(v), sep='', end='' )
            print(',')
        margin=margin[:-2]
        print(margin, '}', sep='', end='')
    else:
        print('${}', end='')

    if not nested:
        print()

init_globals['load'] = configurator

