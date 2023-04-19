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
        parser.add_argument('-l', '--latex', action='store_true', help='Enable LaTeX format')
        parser.add_argument('-o', '--output', nargs='+', default=[], help='output file')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
        try:
            self.exp = self.env.parts.exp[self.opts.exp]
        except Exception:
            print('Unable to retrieve exp '+self.opts.exp)

        self.namespace = self.exp.namespace
        self.context   = self.exp.context
        self.outputs   = self.exp.context.outputs
        try:
            self._juno = self.outputs.observation.juno
        except KeyError:
            self._juno = self.outputs.observation.AD1

        try:
            self._tao = self.outputs.observation_tao
        except KeyError:
            self._tao = None

        self.init_variables()

        self.print_stats()

    def juno(self):
        return self._juno().sum()

    def tao(self):
        if not self._tao:
            return 0.0

        return self._tao().sum()

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

    def print_stats(self):
        data = dict()
        total = data['Total']   = self.juno(), None, self.tao(), None

        ibd_juno = None
        ibd_tao = None
        def add(name, value_juno=None, value_tao=None):
            if value_juno is None:
                value_juno = self.juno()

            if ibd_juno:
                data[name] = value_juno, value_juno/ibd_juno*100.0
            else:
                data[name] = value_juno, 0.0

            if value_tao is None:
                value_tao = self.tao()

            data[name] += value_tao,

            if ibd_tao:
                data[name] += value_tao/ibd_tao*100.0,
            else:
                data[name] += 0.0,

            return value_juno, value_tao

        with self.bkg_all.push_context(0.0):
            add('Reactor+SNF')
            with self.reac['snf_norm'].push_context(0.0):
                ibd_juno, ibd_tao = add('Reactor active')
                with self.reac['offeq'].push_context(0.0):
                    add('Offequilibrium', data['Reactor active'][0] - self.juno(), data['Reactor active'][2] - self.tao())
                    with self.reac['reactor_active_norm'].push_context(0.0):
                        if self.juno()!=0.0:
                            print('Error, nonzero sum', self.juno())

        with self.reac['offeq'].push_context(0.0), self.reac['reactor_active_norm'].push_context(0.0):
            with self.bkg_all.push_context(0.0):
                add('SNF')
                with self.reac['snf_norm'].push_context(0.0):
                    if self.juno()!=0.0:
                        print('Error, nonzero sum', self.juno())
                    for name, par in self.bkg.items():
                        with par.pop_context(0.0):
                            name = name.split('_')[0].capitalize()
                            add(name)

            with self.reac['snf_norm'].push_context(0.0):
                add('Bkg')

        data_t = [ (k,)+v for k,v in data.items() ]

        dt = self._juno.datatype()
        juno1, juno2 = dt.edgesNd[0].front(), dt.edgesNd[0].back()
        if self._tao:
            dt = self._tao.datatype()
            tao1, tao2 = dt.edgesNd[0].front(), dt.edgesNd[0].back()
            taohead =  f'TAO ({tao1:.1f}, {tao2:.1f})'
        else:
            taohead =  'TAO'
        headers=['Name', f'JUNO ({juno1:.1f}, {juno2:.1f})', 'N/reactor active, %', taohead, 'N/reactor active, %']
        options=dict(
            floatfmt='.2f',
            tablefmt='plain'
            )
        if self.opts.latex:
            options['tablefmt']='latex_booktabs'
        t = tabulate(data_t, headers, **options)
        print('JUNO stats')
        print(t)

        tl = None
        for out in self.opts.output:
            with open(out, 'w') as f:
                if out.endswith('.tex'):
                    if tl is None:
                        options['tablefmt']='latex_booktabs'
                        tl = tabulate(data_t, headers, **options)
                    f.write(tl)
                else:
                    f.write(t)
                print('Write output file:', out)





