""" Print JUNO related stats and plot default figures, latex friendly.
    Tweaked for juno_sensitivity_v02
"""
from __future__ import print_function
from gna.ui import basecmd
from collections import OrderedDict
from tools.classwrapper import ClassWrapper
import gna.env
from tabulate import tabulate

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

        self.print_stats()

    def init_variables(self):
        ns = self.namespace

        self.bkg=OrderedDict([(k, NamespaceWrapper(ns[k])) for k in (k0+'_norm' for k0 in ('acc', 'lihe', 'fastn', 'alphan')) if k in ns])
        self.reac=OrderedDict([(k, NamespaceWrapper(ns[k])) for k in (k0+'_norm' for k0 in ('reactor_active', 'snf')) if k in ns])
        self.reac['offeq']=NamespaceWrapper(ns('offeq_scale'))

    def print_stats(self):
        data = OrderedDict()
        total = data['Total']   = self.observation().sum(),

        for p in self.bkg.values(): p.push(0.0)
        data['Reactor+SNF'] = self.observation().sum(),
        self.reac['snf_norm'].push(0.0)
        ibd = data['Reactor only'] = self.observation().sum(),
        self.reac['offeq'].push(0.0)
        data['Offequilibrium'] = data['Reactor only'][0] - self.observation().sum(),
        self.reac['reactor_active_norm'].push(0.0)

        assert self.observation().sum()==0.0
        self.reac['snf_norm'].pop()
        data['SNF'] = self.observation().sum(),

        self.reac['snf_norm'].push(0.0)

        for name, par in self.bkg.iteritems():
            par.pop()
            name = name.split('_')[0].capitalize()
            d = self.observation().sum()
            data[name] = d, d/ibd*100.
            par.push(0.0)

        assert self.observation().sum()==0.0

        for p in self.bkg.values(): p.push(1.0)
        data['Bkg'] = self.observation().sum(),

        data = [ (k,)+v for k,v in data.iteritems() ]

        t = tabulate(data)
        headers=['Name', 'total events', '/reac, %']
        print('JUNO stats')
        print(t, headers)




