"""configurator class allows to load any python file by its filename
and store the contents in a namespace
namespace elements are accessible throught both key access or member acess"""

from __future__ import print_function
import runpy
from os import path
from collections import OrderedDict

class configurator_base(object):
    def __init__(self):
        super(configurator_base, self).__setattr__('__dict__', OrderedDict())

    def __repr__(self):
        return 'configurator'+self.__dict__.__repr__()[11:]

    def get(self, key, default=None):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__dict__.get(key).get( rest, default )

        if isinstance( key, str ) and '.' in key:
            return self.get(key.split('.'))

        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__dict__.__getitem__(key).__getitem__( rest, default )

        if isinstance( key, str ) and '.' in key:
            return self.__getitem__(key.split('.'))

        return self.__dict__.__getitem__(key)

    __getattr__ = __getitem__

    def set(self, key, value):
        if isinstance(value, str):
            if value.startswith('load:'):
                value = configurator(filename = value.replace('load:', ''))
        elif isinstance(value, dict):
            value = configurator(dic=value)

        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__dict__:
                    cfg = self.__dict__[key]=configurator()
                    return cfg.set( rest, value )
                return self.__dict__.get(key).set( rest, value )

        if isinstance( key, str ) and '.' in key:
            return self.set( key.split('.'), value )

        self.__dict__[key] = value

    __setattr__ = set
    __setitem__= set

    def setdefault(self, key, value):
        if isinstance(value, str):
            if value.startswith('load:'):
                value = configurator(filename = value.replace('load:', ''))
        elif isinstance(value, dict):
            value = configurator(dic=value)

        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__dict__:
                    cfg = self.__dict__[key]=configurator()
                    return cfg.setdefault( rest, default )
                return self.__dict__.get(key).setdefault( rest, default )

        if isinstance( key, str ):
            if '.' in key:
                return self.setdefault(key.split('.'), value)
            else:
                return self.__dict__.setdefault(key, value)

        raise Exception( 'Unsupported key type', type(key) )

    def keys(self):
        return self.__dict__.keys()

    def __contains__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                return self.__dict__.get(key).__contains__(rest)

        if isinstance( key, str ):
            if '.' in key:
                return self.__contains__(key.split('.'))
            else:
                return self.__dict__.__contains__(key)

        raise Exception( 'Unsupported key type', type(key) )

    def __call__(self, key):
        if isinstance( key, (list, tuple) ):
            key, rest = key[0], key[1:]
            if rest:
                if not key in self.__dict__:
                    cfg = self.__dict__[key]=configurator()
                    return cfg.__call__( rest )
                return self.__dict__.get(key).__call__(rest)

        if isinstance( key, str ):
            if '.' in key:
                return self.__call__(key.split('.'))
            else:
                if self.__dict__.__contains__( key ):
                    raise Exception( "Can not create nested configuration as the key '%s' already exists"%key )
                value = self.__dict__[key] = configurator()
                return value

        raise Exception( 'Unsupported key type', type(key) )

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
                # if self['@info'].get('verbose', False):
                    # print( 'Skipping nonexistent file', filename )
                continue
            dic = self.__load_dic__(filename, dictonly=True)
            self.__import__(dic)
            unimportant = True

    def __load_dic__(self, filename, dictonly=False):
        print('Loading config file:', filename)
        dic =  runpy.run_path(filename, init_globals={} ) # TODO: add predefined globals

        if dictonly:
            return dic
        return configurator(dic=dic)

    def __import__(self, dic):
        self.__check_for_conflicts__(dic)
        for k, v in sorted(dic.items()):
            if isinstance(k, str) and k.startswith('__'):
                continue
            # if self['@info'].get('verbose', False):
                # if k in self:
                    # print( 'Reset', k, 'to', v.__repr__() )
                # else:
                    # print( 'Set', k, 'to', v.__repr__() )
            self.__setattr__(k, v)

    def __check_for_conflicts__(self, dic):
        """checks whether the dicitonary uses any of forbidden identifiers"""
        forbidden_list  = [ s for s in type(self).__dict__.keys() if not s.startswith('__') ]
        forbidden_items = [ s for s in dic.keys() if s in forbidden_list ]
        if forbidden_items:
            raise Exception("Configuration file '%s' contains following reserved identifiers: %s"%(
                self.__dict__['@info']['@loaded_from'], str(forbidden_items)))

class configurator(configurator_base):
    def __init__(self, filename=None, dic={}, **kwargs):
        super(configurator, self).__init__()

        self.__dict__['@info']={}
        self['@info']['@loaded_from']=filename
        self['@info']['verbose'] = kwargs.pop( 'debug', False )
        if filename:
            self.__load__(filename, **kwargs)
        elif dic:
            self.__import__(dic)


