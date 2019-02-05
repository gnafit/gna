#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

def run_unittests(glb, message='All tests are OK!'):
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print(message)
