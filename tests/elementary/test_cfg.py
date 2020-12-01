#!/usr/bin/env python

from gna.configurator import configurator

cfg = configurator()
cfg.a=1
cfg['b']=2
cfg.set('c', 3)
print( cfg )
print( cfg.a )
print( cfg['a'] )
print( cfg.get('a') )
print()

cfg = configurator()
cfg('b')
cfg['b.d']=1
cfg.b.e=2
cfg.set('f.g',3)
cfg.setdefault('h.g',3)
cfg.setdefault('h.g',4)
print( cfg )
print(cfg.b.d)
print(cfg['b.d'])
print(cfg.get('b.d'))
print(cfg.get('b.d', 'none'))

try:
    print(cfg.get('d.d', 'none'))
    print('\033[32mOK!\033[0m')
except KeyError:
    print('\033[31mFAIL!\033[0m')

try:
    cfg.d
    print('\033[31mFAIL!\033[0m')
except KeyError:
    print('\033[32mOK!\033[0m')

try:
    cfg['d']
    print('\033[31mFAIL!\033[0m')
except KeyError:
    print('\033[32mOK!\033[0m')

try:
    cfg['d.e']
    print('\033[31mFAIL!\033[0m')
except KeyError:
    print('\033[32mOK!\033[0m')
print()

cfg = configurator()
cfg('c.d.e').a=2
print( cfg )
print()

cfg = configurator()
cfg('c.d.e').a= { 5:6 }
cfg.a = dict(test=1, b=2)
cfg.b = { 1:2, 3:4 }
print( cfg )
print()

try:
    cfg.set = 2
    print('\033[32mOK!\033[0m')
except KeyError:
    print('\033[31mFAIL!\033[0m')

try:
    cfg('set')
    print('\033[31mFAIL!\033[0m')
except KeyError:
    print('\033[32mOK!\033[0m')

try:
    cfg('get')
    print('\033[32mOK!\033[0m')
except KeyError:
    print('\033[31mFAIL!\033[0m')
