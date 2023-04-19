""" Print JUNO related stats and plot default figures, latex friendly.
    Tweaked for juno_sensitivity_v02
"""
from gna.ui import basecmd
from tools.classwrapper import ClassWrapper
import gna.env
from tabulate import tabulate
from env.lib.cwd import update_namespace_cwd

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

class ParsPushContext(object):
    def __init__(self, value, pars): self._pars, self._value = pars, value
    def __enter__(self): self._pars.push(self._value)
    def __exit__(self, exc_type, exc_value, traceback): self._pars.pop()

class ParsPopContext(object):
    def __init__(self, value, pars): self._pars, self._value = pars, value
    def __enter__(self): v = self._pars.pop()
    def __exit__(self, exc_type, exc_value, traceback): v = self._pars.push(self._value)

class Pars(object):
    def __init__(self, *pars):
        self._pars=list(pars)
    def push_context(self, value):
        return ParsPushContext(value, self)
    def pop_context(self, value):
        return ParsPopContext(value, self)
    def push(self, value):
        for par in self._pars:
            par.push(value)
    def pop(self):
        for par in self._pars:
            par.pop()
    def __add__(self, other):
        assert isinstance(other, Pars)
        return Pars(*(self._pars+other._pars))

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

    def init_variables(self):
        ns = self.namespace

        self.bkg = {}
        pars_all=[]
        for name in ('acc', 'lihe', 'fastn', 'alphan', 'geonu'):
            pars = []
            for suffix in ('_rate_norm', '_rate_norm_tao'):
                pname = name+suffix

                try:
                    pars.append(ns[pname])
                except KeyError:
                    pass

            pars_all.extend(pars)
            self.bkg[name] = Pars(*pars)

        self.bkg_all = Pars(*pars_all)

        self.reac={k: Pars(NamespaceWrapper(ns[k])) for k in (k0+'_norm' for k0 in ('reactor_active', 'snf')) if k in ns}
        try:
            self.reac['offeq']=Pars(NamespaceWrapper(ns['offeq_scale']))
        except KeyError:
            self.reac['offeq']=Pars(NamespaceWrapper(ns.get_proper_ns('offeq_scale')[0]))

