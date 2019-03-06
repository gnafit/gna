#!/usr/bin/env python
# -*- coding: utf-8 -*-

from printing import *

printl('Global level')
with nextlevel():
    printl('Second level 1')
    printl('Second level 2')
    printl('Second level 3')

    with nextlevel():
        printl('Third level 1')
        printl('Third level 2')
        printl('Third level 3')

    printl('Second level 4')
    printl('Second level 5')
    printl('Second level 6', end=' ')
    printl('again', end=' ')
    printl('and again', end=' ')
    printl()

printl('Global level')

with nextlevel():
    printl('check prefix', prefix='1', end='')
    printl(' [add extra text]')
    printl('check prefix and postfix', prefix='1', postfix='A', end='')
    printl(' [add extra text]')
    printl('check postfix', postfix='A', end='')
    printl(' [add extra text]')
