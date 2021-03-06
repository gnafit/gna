#!/usr/bin/env python

from tools.classwrapper import ClassWrapper
from gna.env import env
import gna.env
import gna.parameters.oscillation

class NamespaceWrapper(ClassWrapper):
    def __new__(cls, obj):
        if not isinstance(obj, gna.env.namespace):
            return obj
        return ClassWrapper.__new__(cls)

    def __init__(self, obj):
        ClassWrapper.__init__(self, obj, NamespaceWrapper)

    def push(self, value):
        for ns in self.walknstree():
            for var in ns.storage.values():
                try:
                    var.push(value)
                except Exception:
                    pass

    def pop(self):
        for ns in self.walknstree():
            for var in ns.storage.values():
                try:
                    var.pop()
                except Exception:
                    pass


ns = env.globalns
gna.parameters.oscillation.reqparameters_reactor(ns('pmns'), dm='23')

ns = NamespaceWrapper(ns)
ns.printparameters()

ns.walknstree()

import IPython; IPython.embed()
