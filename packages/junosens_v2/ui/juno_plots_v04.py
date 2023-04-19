""" Print JUNO related stats and plot default figures, latex friendly.
    Tweaked for juno_sensitivity_v02
"""
from gna.ui import basecmd
from tools.classwrapper import ClassWrapper
import gna.env
from env.lib.cwd import update_namespace_cwd
from mpl_tools.helpers import savefig
from matplotlib import pyplot as plt
import numpy as np
from tools.root_helpers import TFileContext
from os import path
import ROOT as rt
from gna.dispatch import loadcmdclass
from os import path
from junosens_v2.lib.makefcn import MakeFcn

class juno_plots_v04(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('exp', help='JUNO exp instance')
        parser.add_argument('-o', '--output', nargs='+', default=[], help='output file(s)')
        parser.add_argument('-k', '--keep', nargs='+', default=[])

    def init(self):
        update_namespace_cwd(self.opts, 'output')
        try:
            self.exp = self.env.parts.exp[self.opts.exp]
        except Exception:
            print('Unable to retrieve exp '+self.opts.exp)

        self.namespace = self.exp.namespace
        self.context   = self.exp.context
        self.outputs   = self.exp.context.outputs

        self.plot_bump()
        self.plot_oscprob()

    def plot_bump(self):
        kwargs = dict(title='Bump corrections', xlabel='Enu, MeV', ylabel='correction')
        with PlotContext('bump', self.opts, **kwargs) as pc:
            self.outputs.bump_correction_coarse.plot_vs(self.outputs.bump_correction_coarse_centers,
                                                        'o', label='Input')
            self.outputs.bump_correction.plot_vs(self.outputs.enu, 'o',
                                                 markersize=0.5, label='Interpolation', ravel=True)
            pc.ax.legend()

    def plot_oscprob(self):
        Enu = self.outputs.enu
        try:
            Oscprob = self.outputs.oscprob_msw_approx.juno.YJ1
        except KeyError:
            Oscprob = self.outputs.oscprob_matter.juno.YJ1
        rho = self.namespace['rho']
        dm13 = self.namespace['pmns.DeltaMSq13']
        dm12 = self.namespace['pmns.DeltaMSq12']
        # baseline = self.namespace['baseline.juno.YJ1']
        fcn = MakeFcn(Enu, Oscprob)

        NmoSet, _ = loadcmdclass('nmo_set')
        nmoset = NmoSet(self.env, [f'{self.namespace.path}.pmns', '--keep-splitting=13'])
        nmoset.init()

        EnvParsLatex, _ = loadcmdclass('env_pars_latex')
        envparslatex = EnvParsLatex(self.env, [f'{self.namespace.path}.rho', f'{self.namespace.path}.pmns', '-s'])
        if self.opts.output:
            basename = path.basename(path.splitext(self.opts.output[0])[0])
        else:
            basename = None

        enu_edges=np.arange(1.0, 12.0+1.e-6, 0.001, dtype='d')
        enu_centers=0.5*(enu_edges[1:]+enu_edges[:-1])

        # Default
        op_no = fcn(enu_centers)
        if basename:
            envparslatex.opts.output=[f'{basename}_pmns_no.txt']
        envparslatex.init()

        # IO
        nmoset.toggle()
        dm13.push(2.546e-3-dm12.value())
        if basename:
            envparslatex.opts.output=[f'{basename}_pmns_io.txt']
        envparslatex.init()
        op_io = fcn(enu_centers)

        # NO again
        dm13.pop()
        nmoset.toggle()

        # Vacuum and almost vacuum
        rho.push(0.0)
        op_0 = fcn(enu_centers)
        rho.set(1.e-6)
        op_e6 = fcn(enu_centers)

        # Reset
        rho.pop()

        kwargs = dict(title='Osc prob at YJ1', xlabel='Enu, MeV', ylabel='P')
        with PlotContext('oscprob', self.opts, **kwargs) as pc:
            pc.ax.plot(enu_centers, op_no, '-', label='matter NO')
            pc.ax.plot(enu_centers, op_io, '-', label='matter IO')
            pc.ax.plot(enu_centers, op_0, '-', label='vacuum NO')
            pc.ax.plot(enu_centers, op_e6, '--', label='almost vacuum NO', linewidth=2, alpha=0.4)
            pc.ax.legend()

        if self.opts.output:
            fname=path.splitext(self.opts.output[0])[0]+'_oscprob.root'

            with TFileContext(fname, 'recreate') as f:
                for k, t, v in [
                        ('NO_Vacuum', 'NO Vacuum', op_0),
                        ('NO_1e-6',   'NO Vacuum, almost', op_e6),
                        ('NO_2.45',   'NO Matter', op_no),
                        ('IO_2.45',   'IO Matter', op_io),
                        ]:
                    h=rt.TH1D(k, t, enu_centers.size, enu_edges[0], enu_edges[-1])
                    b = h.get_buffer()
                    c = h.GetXaxis().get_bin_centers()
                    assert np.allclose(c, enu_centers, atol=1.e-14, rtol=0)
                    b[:]=v
                    h.SetEntries(v.sum())

                    f.WriteTObject(h, k, 'overwrite')

class PlotContext(object):
    def __init__(self, name, opts, *args, **kwargs):
        self._name = name
        self._opts = opts
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        self._fig = plt.figure(self._name)
        self._ax = plt.subplot(111, *self._args, **self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        close = not self._name in self._opts.keep
        savefig(self._opts.output, suffix=f'_{self._name}', close=close)

    @property
    def ax(self):
        return self._ax

    @property
    def fig(self):
        return self._fig

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


