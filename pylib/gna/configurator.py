# -*- coding: utf-8 -*-

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

def process_key(key):
    listkey = None
    if isinstance( key, basestring ):
        if '.' in key:
            listkey = list(key.split('.'))
    elif isinstance( key, (list, tuple) ):
        listkey = []
        for sk in key:
            if isinstance(sk, (list, tuple)):
                listkey+=list(sk)
            else:
                listkey.append(sk)

    if listkey:
        return listkey[0], listkey[1:]

    return key, None

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
            self.__import__(OrderedDict(sorted(kwargs.items())))

    def __repr__(self):
        return self.__storage__.__repr__().replace('Ordered', 'Nested', 1)

    def __str__(self, margin=''):
        if not self.__bool__():
            return '${}'

        res='${\n'
        margin+='  '
        for k, v in self.items():
            res+='{margin}{key} : '.format( margin=margin, key=k )
            if isinstance( v, NestedDict ):
                res+=v.__str__(margin)
            elif isinstance( v, basestring ):
                res+=repr(v)
            else:
                res+=str(v)
            res+=',\n'
        margin=margin[:-2]

        return res+margin+'}'

    def __bool__(self):
        return bool(self.keys())

    def __len__(self):
        return len(self.keys())

    def _set_parent(self, parent):
        super(NestedDict, self).__setattr__('__parent__', parent)

    def parent(self, n=1):
        if n==1:
            return self.__parent__

        if n==0:
            return self

        if n<0:
            raise Exception('Invalid parent depth')

        if self.__parent__ is None:
            raise Exception('No parent')

        return self.__parent__.parent( n-1 )


    def parent_key(self):
        if self.__parent__ is None:
            return None

        for k, v in self.__parent__.items():
            if v is self:
                return k

        raise KeyError( "Failed to determine own key in the parent dictionary" )

    def get(self, key, *args, **kwargs):
        key, rest=process_key(key)
        if rest:
            sub = self.__storage__.get(key)
            if sub is None:
                if args:
                    return args[0]
                raise KeyError( "No nested key '%s'"%key )
            return sub.get( rest, *args, **kwargs )

        types=kwargs.pop('types', None)
        obj=self.__storage__.get(key, *args, **kwargs)
        if types:
            if not isinstance(obj, types):
                if isinstance(types, tuple):
                    raise Exception('The field "{}" is expected to be of one of types {}, not {}'.format(key, str([t.__name__ for t in types]), type(obj).__name__))
                else:
                    raise Exception('The field "{}" is expected to be of type {}, not {}'.format(key, types.__name__, type(obj).__name__))
        return obj

    def __getitem__(self, key):
        key, rest=process_key(key)
        if rest:
            return self.__storage__.__getitem__(key).__getitem__( rest )

        return self.__storage__.__getitem__(key)

    __getattr__ = __getitem__

    def set(self, key, value, loading_from_file=False ):
        if isinstance(value, dict):
            value = NestedDict(value)
        if isinstance(value, NestedDict):
            value._set_parent( self )

        key, rest=process_key(key)
        if rest:
            if not key in self.__storage__:
                cfg = self.__storage__[key]=NestedDict()
                cfg._set_parent( self )
                return cfg.set( rest, value )
            return self.__storage__.get(key).set( rest, value )

        self.__storage__[key] = value

    __setattr__ = set
    __setitem__= set

    def setdefault(self, key, value):
        if isinstance(value, dict):
            value = NestedDict(value)
        if isinstance(value, NestedDict):
            value._set_parent( self )

        key, rest=process_key(key)
        if rest:
            if not key in self.__storage__:
                cfg = self.__storage__[key]=NestedDict()
                cfg._set_parent( self )
                return cfg.setdefault( rest, value )
            return self.__storage__.get(key).setdefault( rest, value )

        return self.__storage__.setdefault(key, value)

    def keys(self):
        return self.__storage__.keys()

    def __iter__(self):
        return iter(self.__storage__)

    def values(self):
        return self.__storage__.values()

    def items(self):
        return self.__storage__.items()

    def __contains__(self, key):
        key, listkey=process_key(key)

        key, rest=process_key(key)
        if rest:
            return self.__storage__.get(key).__contains__(rest)

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
            if isinstance(k, basestring) and k.startswith('__'):
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

class uncertain(object):
    def __init__(self, central, uncertainty=None, mode='fixed'):
        assert mode in ['absolute', 'relative', 'percent', 'fixed'], 'Unsupported uncertainty mode '+mode

        assert (mode=='fixed')==(uncertainty is None), 'Inconsistent mode and uncertainty'

        if mode=='percent':
            mode='relative'
            uncertainty*=0.01

        if mode=='relative':
            assert central!=0, 'Central value should differ from 0 for relative uncertainty'

        self.central = central
        self.uncertainty   = uncertainty
        self.mode    = mode

    def __str__(self):
        res = '{central:.6g}'.format(central=self.central)

        if self.mode=='fixed':
            return res

        if self.mode=='relative':
            sigma    = self.central*self.uncertainty
            relsigma = self.uncertainty
        else:
            sigma    = self.uncertainty
            relsigma = sigma/self.central

        res +=( '±{sigma:.6g}'.format(sigma=sigma) )

        if self.central:
            res+=( ' [{relsigma:.6g}%]'.format(relsigma=relsigma*100.0) )

        return res

    def __repr__(self):
        return 'uncertain({central!r}, {uncertainty!r}, {mode!r})'.format( **self.__dict__ )

def uncertaindict(*args, **kwargs):
    common = dict(
        central = kwargs.pop( 'central', None ),
        uncertainty   = kwargs.pop( 'uncertainty',   None ),
        mode    = kwargs.pop( 'mode',    None ),
    )
    missing = [ s for s in ['central', 'uncertainty', 'mode'] if not common[s] ]
    res  = OrderedDict( *args, **kwargs )

    for k, v in res.items():
        kcommon = common.copy()
        if isinstance(v, dict):
            kcommon.update( v )
        else:
            if isinstance( v, (int, float) ):
                v = (v, )
            kcommon.update( zip( missing, v ) )

        res[k] = uncertain( **kcommon )

    return res

init_globals['load'] = configurator
init_globals['uncertain'] = uncertain
init_globals['uncertaindict'] = uncertaindict

