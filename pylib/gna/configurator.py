"""configurator class allows to load any python file by its filename
and store the contents in a namespace
namespace elements are accessible throught both key access or member acess"""

from __future__ import print_function
import runpy
from os import path
from collections import OrderedDict
from weakref import WeakKeyDictionary

meta = WeakKeyDictionary()
init_globals = dict( percent=0.01 )
forbidden_keys = []

class NestedDict(object):
    def __init__(self, iterable=None, **kwargs):
        super(NestedDict, self).__setattr__('__dict__', OrderedDict())

        meta[self] = dict()

        if iterable:
            self.__import__(OrderedDict(iterable))

        if kwargs:
            self.__import__(kwargs)

    def __repr__(self):
        return 'NestedDict'+self.__dict__.__repr__()[11:]

    def get(self, key, default=None):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                sub = self.__dict__.get(key)
                if sub is None:
                    raise KeyError( "No nested key '%s'"%key )
                return sub.get( rest, default )

        if isinstance( key, basestring ) and '.' in key:
            return self.get(key.split('.'))

        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__dict__.__getitem__(key).__getitem__( rest )

        if isinstance( key, basestring ) and '.' in key:
            return self.__getitem__(key.split('.'))

        return self.__dict__.__getitem__(key)

    __getattr__ = __getitem__

    def set(self, key, value):
        if isinstance(value, basestring):
            if value.startswith('load:'):
                value = configurator(filename = value.replace('load:', ''))
        elif isinstance(value, dict):
            value = NestedDict(value)

        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__dict__:
                    cfg = self.__dict__[key]=NestedDict()
                    return cfg.set( rest, value )
                return self.__dict__.get(key).set( rest, value )

        if isinstance( key, basestring ) and '.' in key:
            return self.set( key.split('.'), value )

        if key in forbidden_keys:
            raise KeyError( "Can not use key '%s' due to technical limitations"%key )

        self.__dict__[key] = value

    __setattr__ = set
    __setitem__= set

    def setdefault(self, key, value):
        if isinstance(value, basestring):
            if value.startswith('load:'):
                value = configurator(filename = value.replace('load:', ''))
        elif isinstance(value, dict):
            value = NestedDict(value)

        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__dict__:
                    cfg = self.__dict__[key]=NestedDict()
                    return cfg.setdefault( rest, value )
                return self.__dict__.get(key).setdefault( rest, value )

        if isinstance( key, basestring ):
            if '.' in key:
                return self.setdefault(key.split('.'), value)

        if key in forbidden_keys:
            raise KeyError( "Can not use key '%s' due to technical limitations"%key )

        return self.__dict__.setdefault(key, value)

    def keys(self):
        return self.__dict__.keys()

    def __contains__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__dict__.get(key).__contains__(rest)

        if isinstance( key, basestring ):
            if '.' in key:
                return self.__contains__(key.split('.'))

        return self.__dict__.__contains__(key)

    def __call__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__dict__:
                    cfg = self.__dict__[key]=NestedDict()
                    return cfg.__call__( rest )
                return self.__dict__.get(key).__call__(rest)

        if isinstance( key, basestring ):
            if '.' in key:
                return self.__call__(key.split('.'))

        if key in forbidden_keys:
            raise KeyError( "Can not use key '%s' due to technical limitations"%key )

        if self.__dict__.__contains__( key ):
            raise KeyError( "Can not create nested configuration as the key '%s' already exists"%key )

        value = self.__dict__[key] = NestedDict()
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
            del dic[k]

        if dictonly:
            return dic
        return NestedDict(dic)

    def __import__(self, dic):
        self.__check_for_conflicts__(dic)
        for k, v in sorted(dic.items()):
            if isinstance(k, str) and k.startswith('__'):
                continue
            if meta[self].get('verbose', False):
                if k in self:
                    print( 'Reset', k, 'to', v.__repr__() )
                else:
                    print( 'Set', k, 'to', v.__repr__() )
            self.__setattr__(k, v)

    def __check_for_conflicts__(self, dic):
        """checks whether the dicitonary uses any of forbidden identifiers"""
        forbidden_items = [ s for s in dic.keys() if s in forbidden_keys or s in init_globals ]
        if forbidden_items:
            raise KeyError("Configuration file '%s' contains following reserved identifiers: %s"%(
                self.__dict__.get('@loaded_from', ''), basestring(forbidden_items)))

forbidden_keys = [ s for s in NestedDict.__dict__.keys() if not s.startswith('__') ]

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


