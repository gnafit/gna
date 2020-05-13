"""Print JUNO related stats and plot default figures, latex friendly.
   Tweaked for juno_sensitivity_v02
"""
from __future__ import print_function
from gna.ui import basecmd
from collections import OrderedDict
from tools.classwrapper import ClassWrapper

class NamespaceWrapper(ClassWrapper):
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

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('exp', help='JUNO exp instance')

    def init(self):
        try:
            self.exp = self.env.parts.exp[self.opts.exp]
        except Exception:
            print('Unable to retrieve exp '+self.opts.exp)

        self.namespace = self.exp.namespace
        self.context   = self.exp.context
        self.outputs   = self.exp.context.outputs
        self.observation = self.outputs.observation.AD1

        self.init_variables()

    def init_variables(self):
        ns = self.namespace

        self.bkg=OrderedDict([(k, ns[k]) for k in (k0+'_norm' for k0 in ('acc', 'lihe', 'fastn', 'alphan')) if k in ns])
        self.reac=OrderedDict([(k, ns[k]) for k in (k0+'_norm' for k0 in ('reactor_active', 'snf')) if k in ns])
        self.reac['offeq']=ns('offeq_scale')

