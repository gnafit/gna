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
    def __new__(cls, obj, *args, **kwargs):
        if not isinstance(obj, gna.env.namespace):
            return obj
        return ClassWrapper.__new__(cls)

    def __init__(self, obj, parent=None):
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
        parser.add_argument('-l', '--latex', action='store_true', help='Enable LaTeX format')
        parser.add_argument('-o', '--output', nargs='+', default=[], help='output file')

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

        self.bkg=OrderedDict([(k, NamespaceWrapper(ns[k])) for k in (k0+'_rate_norm' for k0 in ('acc', 'lihe', 'fastn', 'alphan', 'geonu')) if k in ns])
        self.reac=OrderedDict([(k, NamespaceWrapper(ns[k])) for k in (k0+'_norm' for k0 in ('reactor_active', 'snf')) if k in ns])
        try:
            self.reac['offeq']=NamespaceWrapper(ns['offeq_scale'])
        except KeyError:
            self.reac['offeq']=NamespaceWrapper(ns.get_proper_ns('offeq_scale')[0])

    def print_stats(self):
        data = OrderedDict()
        total = data['Total']   = self.observation().sum(),

        ibd = None
        def add(name, value=None):
            if value is None:
                value = self.observation().sum()
            if ibd:
                data[name] = value, value/ibd*100.0
            else:
                data[name] = value,
            return value

        for p in self.bkg.values(): p.push(0.0)
        add('Reactor+SNF')
        self.reac['snf_norm'].push(0.0)
        ibd = add('Reactor active')
        self.reac['offeq'].push(0.0)
        add('Offequilibrium', data['Reactor active'][0] - self.observation().sum())
        self.reac['reactor_active_norm'].push(0.0)

        if self.observation().sum()!=0.0:
            print('Error, nonzero sum', self.observation().sum())

        self.reac['snf_norm'].pop()
        add('SNF')

        self.reac['snf_norm'].push(0.0)

        for name, par in self.bkg.iteritems():
            par.pop()
            name = name.split('_')[0].capitalize()
            add(name)
            par.push(0.0)

        if self.observation().sum()!=0.0:
            print('Error, nonzero sum', self.observation().sum())

        for p in self.bkg.values(): p.pop()
        add('Bkg')

        # Revert
        self.reac['snf_norm'].pop()
        self.reac['offeq'].pop()
        self.reac['reactor_active_norm'].pop()

        data = [ (k,)+v for k,v in data.iteritems() ]

        headers=['Name', 'total events', 'N/reactor active, %']
        options=dict(
            floatfmt='.2f',
            tablefmt='plain'
            )
        if self.opts.latex:
            options['tablefmt']='latex_booktabs'
        t = tabulate(data, headers, **options)
        print('JUNO stats')
        print(t)

        tl = None
        for out in self.opts.output:
            with open(out, 'w') as f:
                if out.endswith('.tex'):
                    if tl is None:
                        options['tablefmt']='latex_booktabs'
                        tl = tabulate(data, headers, **options)
                    f.write(tl)
                else:
                    f.write(t)
                print('Write output file:', out)





