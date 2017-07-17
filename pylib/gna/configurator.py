"""configurator class allows to load any python file by its filename
and store the contents in a namespace
namespace elements are accessible throught both key access or member acess"""

from __future__ import print_function
import runpy
from os import path

class configurator_base(object):
    __verbose__ = False
    # def __init__(self):
        # pass

    def get(self, name, default=None):
        return self.__dict__.get(name, default)

    def set(self, key, value):
        self.__dict__[key] = value

    def setdefault(self, key, value):
        return self.__dict__.setdefault(key, value)

    def __contains__(self, name):
        return self.__dict__.__contains__(name)

    def __setattr__(self, key, value):
        if type(value)==str:
            if value.startswith('load:'):
                value = configurator(filename = value.replace('load:', ''))
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        if type(value)==str:
            if value.startswith('load:'):
                value = configurator(filename = value.replace('load:', ''))
        self.__dict__[key] = value

    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except:
            raise Exception( "Config file '%s' doesn't define key <%s>"%( self.__dict__['@info']['@loaded_from'], key ) )

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
                if self.__verbose__:
                    print( 'Skipping nonexistent file', filename )
                continue
            dic = self.__load_dic__(filename, dictonly=True)
            self.__import__(dic)
            unimportant = True

    def __load_dic__(self, filename, dictonly=False):
        print('Loading config file:', filename)
        dic =  runpy.run_path(filename, init_globals={}) # TODO: add predefined globals

        if dictonly:
            return dic
        return configurator(dic=dic)

    def __import__(self, dic):
        self.__check_for_conflicts__(dic)
        for k, v in sorted(dic.items()):
            if k.startswith('__'):
                continue
            if self.__verbose__:
                if k in self:
                    print( 'Reset', k, 'to', v.__repr__() )
                else:
                    print( 'Set', k, 'to', v.__repr__() )
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
        # super(configurator, self).__init__()

        self['@info']={'@loaded_from': filename}
        self.__verbose__ = kwargs.pop( 'debug', False )
        if filename:
            self.__load__(filename, **kwargs)
        elif dic:
            self.__import__(dic)


