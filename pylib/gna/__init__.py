# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import os.path

from load import ROOT as R

def datapath(fname):
    basedir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(basedir, fname)

class ConstructorsWrapper(object):
    def __init__(self):
        from gna import constructors, context
        self.__constructors = constructors
        self.__context = context
        self.__chain = [self.__findwrapper, self.__findtemplate, self.__findclass]
        self.__notfound=['notfound']

    def __getattr__(self, name):
        ret = self.__notfound
        for finder in self.__chain:
            try:
                # print('try', finder, name)
                ret, save = finder(name)
                # print('found', ret, save)
            except AttributeError as e:
                # print('  fail', e)
                pass
            else:
                # print('  found')
                break

        if ret is self.__notfound:
            raise Exception('Do not know GNAObject '+name)

        if save and not isinstance(ret, str):
            setattr(self, name, ret)

        return ret

    def __findwrapper(self, name):
        return getattr(self.__constructors, name), True

    def __findtemplate(self, name):
        template = getattr(R.GNA.GNAObjectTemplates, name+'T')
        cls = template(self.__context.current_precision())
        return cls, False

    def __findclass(self, name):
        return getattr(R, name), True

constructors = ConstructorsWrapper()
