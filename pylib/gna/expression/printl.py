#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

printlevel = 0
class nextlevel():
    def __enter__(self):
        global printlevel
        printlevel+=1

    def __exit__(self, *args, **kwargs):
        global printlevel
        printlevel-=1

def printl(*args, **kwargs):
    prefix = kwargs.pop('prefix', ())

    if prefix:
        print( *prefix, end='' )
    print('    '*printlevel, sep='', end='')
    print(*args, **kwargs)

debugmethods=False
if debugmethods:
    def methodname(fcn):
        def newfcn(self, *args, **kwargs):
            printl('methodname', type(self).__name__, getattr(self, 'name', '?'), fcn.__name__, *args, **kwargs)
            with nextlevel():
                return fcn(self, *args, **kwargs)
        return newfcn
else:
    def methodname(fcn):
        return fcn
