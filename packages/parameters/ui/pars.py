"""Manage parameters."""

from gna.ui import basecmd
from tools.classwrapper import ClassWrapper
from gna import env
import numpy as np
import ROOT as R
from collections.abc import Mapping

GaussianParameter = R.GaussianParameter('double')

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('root', nargs='*', default=[], help='the root namespace to work with')
        parser.add_argument('-g', '--group', nargs='*', default=[], help='parameter groups to work with')

        parser.add_argument('--free', action='store_true', help='set_free')

        ff = parser.add_mutually_exclusive_group()
        ff.add_argument('--variable', action='store_true', help='set as not fixed')
        ff.add_argument('--fix', '--fixed', action='store_true', help='set fixed')

        randomize = parser.add_argument_group('rand', description='Randomization')
        randomize.add_argument('--randomize-constrained', action='store_true', help='randomize normally distributed constrained parameters')
        randomize.add_argument('--randomize-free', type=float, help='randomize normally distributed free parameters accroding to a given width')

        setting = parser.add_mutually_exclusive_group()
        setting.add_argument('--pop', action='store_true', help='restore the previous value')
        setting.add_argument('--push', type=float, help='push the value and backup the previous one')

        sigma = parser.add_mutually_exclusive_group()
        sigma.add_argument('--sigma', type=float, help='set sigma')
        sigma.add_argument('--relsigma', type=float, help='set relativesigma')

        parser.add_argument('--correlation', type=float, help='set correlation between parameters within each group independently')

        # parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

    def init(self):
        self.namespaces=[NamespaceWrapper(self.env.globalns(root)) for root in self.opts.root]
        if self.opts.group:
            groups_loc = self.env.future['parameter_groups']
            self.namespaces.extend([NamespaceWrapper(groups_loc[name]) for name in self.opts.group])
        if not self.namespaces:
            self.namespaces = [NamespaceWrapper(self.env.globalns)]

        if self.opts.fix:
            for ns in self.namespaces:
                ns.setFixed(True)
        elif self.opts.variable:
            for ns in self.namespaces:
                ns.setFixed(False)
        if self.opts.free:
            for ns in self.namespaces:
                ns.setFree(True)
        if self.opts.push is not None:
            for ns in self.namespaces:
                ns.push(self.opts.push)
        if self.opts.sigma is not None:
            for ns in self.namespaces:
                ns.setSigma(self.opts.sigma)
        if self.opts.relsigma is not None:
            for ns in self.namespaces:
                ns.setRelSigma(self.opts.sigma)
        elif self.opts.pop:
            for ns in self.namespaces:
                ns.pop()
        elif self.opts.correlation:
            for ns in self.namespaces:
                ns.setCorrelation(self.opts.correlation)
        if self.opts.randomize_constrained:
            for ns in self.namespaces:
                ns.setRandomConstrained()
        if self.opts.randomize_free:
            for ns in self.namespaces:
                ns.setRandomFree(self.opts.randomize_free)

class NamespaceWrapper(ClassWrapper):
    def __new__(cls, obj, *args, **kwargs):
        if not isinstance(obj, (env.namespace, ClassWrapper, Mapping)):
            return obj
        return ClassWrapper.__new__(cls)

    def __init__(self, obj, parent=None):
        if isinstance(obj, ClassWrapper):
            obj=obj.unwrap()
        ClassWrapper.__init__(self, obj, types=(NamespaceWrapper, dict))

    def setFixed(self, f):
        self._apply(lambda v: v.setFixed(f))

    def setFree(self, f):
        self._apply(lambda v: v.setFree(f))

    def setCorrelation(self, c):
        from itertools import combinations
        for p1, p2 in combinations(self._walk(), 2):
            p1.setCorrelation(p2, c)

    @staticmethod
    def _setRandomConstrained(v):
        if not isinstance(v, GaussianParameter):
            return
        if v.isFixed() or v.isFree():
            return
        v.setNormalValue(np.random.normal())

    def setRandomConstrained(self):
        self._apply(self._setRandomConstrained)

    def setRandomFree(self, sigma):
        def fcn(v):
            if v.isFixed() or not v.isFree():
                return
            v.set(v.central()+np.random.normal(scale=sigma))

        self._apply(fcn)

    def pop(self):
        self._apply(lambda v: v.pop())

    def push(self, value=None):
        if value is None:
            fcn=lambda v: v.push()
        else:
            fcn=lambda v: v.push(value)

        self._apply(fcn)

    def setSigma(self, sigma: float) -> None:
        self._apply(lambda v: v.setSigma(sigma))

    def setRelSigma(self, relsigma: float) -> None:
        self._apply(lambda v: v.setRelSigma(relsigma))

    def _apply(self, fcn):
        for var in self._walk():
            fcn(var)

    def _walk(self):
        if isinstance(self._obj, Mapping):
            yield from self.values()
        else:
            for ns in self.walknstree():
                yield from ns.storage.values()


