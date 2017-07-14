"""config class allows to load any python file by its filename
and store the contents in a namespace
namespace elements are accessible throught both key access or member acess"""

from __future__ import print_function
import runpy

class config(object):
    def __init__(self, filename=None, dic={}, **kwargs):
        self['@info']={'@loaded_from': filename}
        self.__verbose__ = kwargs.pop( 'debug', False )
        if filename:
            self.__load__(filename)
        elif dic:
            self.__import__(dic)

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
                value = config(filename = value.replace('load:', ''))
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        if type(value)==str:
            if value.startswith('load:'):
                value = config(filename = value.replace('load:', ''))
        self.__dict__[key] = value

    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except:
            raise Exception( "Config file '%s' doesn't define key <%s>"%( self.__dict__['@info']['@loaded_from'], key ) )

    def __load__(self, filename):
        dic = self.__load_dic__(filename, dictonly=True)
        self.__import__(dic)

    def __load_dic__(self, filename, dictonly=False):
        print('Loading config file:', filename)
        dic =  runpy.run_path(filename, init_globals={}) # TODO: add predefined globals

        if dictonly:
            return dic
        return config(dic=dic)

    def __import__(self, dic):
        self.__check_for_conflicts__(dic)
        for k, v in sorted(dic.items()):
            if k.startswith('__'):
                continue
            if self.__verbose__:
                if k in self:
                    print( 'Reset', k, 'to', v )
                else:
                    print( 'Set', k, 'to', v )
            self.__setattr__(k, v)

    def __check_for_conflicts__(self, dic):
        """checks whether the dicitonary uses any of forbidden identifiers"""
        forbidden_list  = [ s for s in type(self).__dict__.keys() if not s.startswith('__') ]
        forbidden_items = [ s for s in dic.keys() if s in forbidden_list ]
        if forbidden_items:
            raise Exception("Configuration file '%s' contains following reserved identifiers: %s"%(
                self.__dict__['@info']['@loaded_from'], str(forbidden_items)))

