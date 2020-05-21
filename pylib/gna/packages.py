# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import os

def iterate_module_paths(name):
    paths = os.environ.get('GNAPATH', './packages').split(':')
    for path in paths:
        try:
            for sub in sorted(os.listdir(path)):
                sub = '{path}/{sub}'.format(path=path, sub=sub)
                if not os.path.isdir(sub):
                    continue
                sub = '{sub}/{name}'.format(sub=sub, name=name)
                if os.path.isdir(sub):
                    yield sub
        except Exception:
            pass

if __name__ == "__main__":
    print(list(iterate_module_paths('ui')))
    print(list(iterate_module_paths('experiments')))
