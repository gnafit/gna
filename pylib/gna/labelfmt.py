# -*- coding: utf-8 -*-

import string
debug = False

labels = dict(
    logL1_label = 'agreement to data',
    logL1       = r'$\log L_1 = \log \chi^2$',
    logL2_label = 'regularity',
    logL2       = r'$\log L_2/\tau$',
    logtau      = r'$\log \tau$'
)

class LFormatter(string.Formatter):
    def get_value( self, key, args, kwargs ):
        # no positional arguments are supported
        # if type(key) is long:
            # return args[key]

        if key in kwargs:
            return kwargs[key]

        if key.startswith( '^' ):
            return kwargs[key[1:]].capitalize()

        if key.startswith( '$' ):
            return self.get_value( labels[key[1:]], args, kwargs )

        return key

    def __call__( self, s ):
        return self.format( s, **labels )

formatter = LFormatter()


