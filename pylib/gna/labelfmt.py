# -*- coding: utf-8 -*-

from __future__ import print_function
import string
from collections import OrderedDict
import itertools as I

class ndict(object):
    def __init__(self, dicts=None):
        self.dicts = OrderedDict()

        if dicts:
            self.dicts.update( dicts )

    def get(self, k, kwargs):
        for i, d in enumerate(I.chain((kwargs,), reversed(self.dicts.values()))):
            if not d:
                continue
            res = d.get(k, None)
            if res:
                return res
        return None

    def register( self, label, dict ):
        self.dicts[label] = dict

dictionaries = ndict()
def reg_dictionary( label, dict ):
    dictionaries.register(label, dict)

labels = dict(
    logL1_label = 'agreement to data',
    logL1       = r'$\log L_1 = \log \chi^2$',
    logL2_label = 'regularity',
    logL2       = r'$\log L_2/\tau$',
    logtau      = r'$\log \tau$'
)
reg_dictionary( 'unfolding', labels )

class LFormatter(string.Formatter):
    def get_value( self, key, args, kwargs ):
        res = dictionaries.get(key, kwargs)
        if res:
            return res

        if key.startswith( '$' ):
            return self.get_value( dictionaries.get(key[1:], kwargs), args, kwargs )

        if key.startswith( '^' ):
            return (self.get_value( key[1:], args, kwargs )).capitalize()

        return '?'+key+'?'

    def __call__( self, s, **kwargs ):
        return self.format( s, **kwargs )

formatter = LFormatter()


