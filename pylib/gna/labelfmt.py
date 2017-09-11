# -*- coding: utf-8 -*-

from __future__ import print_function
import string
from collections import OrderedDict
import itertools as I

from gna.configurator import configurator
cfg = configurator( '{location}/gna/labels.py', subst='default', debug=False )

class LFormatter(string.Formatter):
    def get_value( self, key, args, kwargs ):
        if isinstance( key, (int, long) ):
            return args[key]

        res = dictionaries.get(key, kwargs)
        if not res is None:
            return res

        if key.startswith( '$' ):
            return self.get_value( dictionaries.get(key[1:], kwargs), args, kwargs )

        if key.startswith( '^' ):
            return (self.get_value( key[1:], args, kwargs )).capitalize()

        return '?'+key+'?'

    def __call__( self, s, *args, **kwargs ):
        return self.format( s, *args, **kwargs )

    def s( self, key, **kwargs ):
        return self( '{%s}'%key, **kwargs )

    def u( self, var, offset=None, **kwargs ):
        label = self( '{%s}'%var, **kwargs )
        if var.startswith( '^' ):
            var = var[1:]
        unit  = self( '{%s_unit}'%var, **kwargs )

        power = cfg.common_offsets.get( var, None ) if offset is None else offset
        if power:
            offset = cfg.offset.format( power=power )

            if unit:
                return cfg.name_offset_unit.format( name=label, unit=unit, offset=offset )

            return cfg.name_offset.format( name=label, offset=offset )

        if unit:
            return cfg.name_unit.format( name=label, unit=unit )

        return label

class ndict(object):
    def __init__(self, dicts=None):
        self.dicts = OrderedDict()

        if dicts:
            self.dicts.update( dicts )

    def get(self, k, kwargs):
        for i, d in enumerate(I.chain((kwargs,), reversed(self.dicts.values()))):
            if not d:
                continue
            # print( i, k )
            res = d.get(k, None)
            if not res is None:
                # print( '  ->', res )
                return res
        return None

    def register( self, label, dict ):
        self.dicts[label] = dict

formatter = LFormatter()
dictionaries = ndict()
def reg_dictionary( label, dict ):
    dictionaries.register(label, dict)

def load_dictionaries( path ):
    from os import listdir
    from gna.configurator import configurator
    for filename in listdir( path ):
        if not filename.endswith('.py'):
            continue
        filepath = path+'/'+filename
        cfg = configurator( filepath )
        reg_dictionary( filename[:-2], cfg.dictionary )

load_dictionaries( './config/dictionaries' )



