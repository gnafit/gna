# -*- coding: utf-8 -*-

from __future__ import print_function

printlevel = 0
printmargin = '    '

class nextlevel():
    def __enter__(self):
        global printlevel
        printlevel+=1

    def __exit__(self, *args, **kwargs):
        global printlevel
        printlevel-=1

def current_level():
    return printlevel

def printl(*args, **kwargs):
    prefix = kwargs.pop('prefix', ())

    if prefix:
        print( *prefix, end='' )

    print(printmargin*printlevel, sep='', end='')
    print(*args, **kwargs)


