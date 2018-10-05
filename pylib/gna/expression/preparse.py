#!/usr/bin/env python
# -*- coding: utf-8 -*-

def balanced_pair(s, start=0):
    level = 0
    for i, c in enumerate(s[start:]):
        if c=='(':
            level+=1
        elif c==')':
            level-=1

        if level<0:
            break
    else:
        return len(s)

    return start+i

def open_fcn( s ):
    s = s.replace('\n','')
    while '|' in s:
        start = s.index('|')
        end = balanced_pair( s, start )

        pre, mid, post = s[:start], s[start+1:end], s[end:]
        s=''.join( [ pre, '(', mid, ')', post ] )

    return s


