# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import NestedDict
from gna.expression.preparse import open_fcn
from gna.expression.operation import *
from gna.env import env
import re
import inspect

class VTContainer_v01(OrderedDict):
    _order=None
    def __init__(self, *args, **kwargs):
        super(VTContainer_v01, self).__init__(*args, **kwargs)

    def set_indices(self, indices):
        self._order=indices.order

    def __missing__(self, key):
        newvar = Variable(key, order=self._order)
        self[key] = newvar
        return newvar

    def __setitem__(self, key, value):
        if isinstance(value, Indexed):
            if value.name is undefinedname and key!='__tree__':
                value.name = key
            value.nindex.arrange(self._order)
            value.expandable=False
        elif inspect.isclass(value) and issubclass(value, Operation):
            value.order=self._order

        OrderedDict.__setitem__(self, key, value)
        return value

class Expression_v01(object):
    operations = dict(sum=OSum, prod=OProd, concat=OConcat, accumulate=Accumulate, bracket=bracket, inverse=OInverse )
    tree = None
    def __init__(self, expression, indices=[], **kwargs):
        if isinstance(expression, basestring):
            self.expressions_raw = [expression]
        elif isinstance(expression, (tuple, list)):
            self.expressions_raw = list(expression)
        else:
            raise Exception('Unsupported expression: {!r}'.format(expression))

        cexpr = re.compile('\s*#.*')
        rexpr = re.compile('\n\s+')
        self.expressions_raw = [ rexpr.sub('', cexpr.sub('', e)) for e in self.expressions_raw ]
        self.expressions = [open_fcn(expr) for expr in self.expressions_raw]

        self.globals=VTContainer_v01()
        self.defindices(indices, **kwargs)
        self.set_operations()

    def set_operations(self):
        for name, op in self.operations.iteritems():
            self.globals[name]=op

    def parse(self):
        if self.tree:
            raise Exception('Expression is already parsed')

        self.trees = []
        for expr in self.expressions:
            texpr = '__tree__ = '+expr
            try:
                exec(texpr, self.globals, self.globals)
                tree = self.globals.pop('__tree__')
            except:
                print('Failed to evaluate expression:')
                print(expr)
                raise
            self.trees.append(tree)

        self.tree=self.trees[-1]

    def guessname(self, ilib, *args, **kwargs):
        lib = dict()
        for k, v in ilib.items():
            v['name'] = k
            exprs = v['expr']
            if isinstance(exprs, str):
                exprs=[exprs]
            for expr in exprs:
                lib[expr] = v
        for tree in self.trees:
            tree.guessname(lib, *args, **kwargs)

    def __str__(self):
        return self.expressions_raw

    def __repr__(self):
        return 'Expression("{}")'.format(self.expressions_raw)

    def defindices(self, defs):
        if isinstance(defs, NIndex):
            self.nindex=defs
        else:
            self.nindex = NIndex(fromlist=defs)

        for short, idx in self.nindex.indices.items():
            self.globals[short] = idx

            slave=idx.slave
            if slave:
                self.globals[slave.short]=slave
        self.globals.set_indices(self.nindex)

    def build(self, context):
        if not self.tree:
            raise Exception('Expression is not initialized, call parse() method first')

        context.set_indices(self.nindex)
        for tree in self.trees:
            creq = tree.require(context)

        context.build_bundles()

        with context:
            for tree in self.trees:
                tree.bind(context)

class ItemProvider(object):
    """Container for the bundle class, bundle configuration and provided items"""
    bundle=None
    def __init__(self, cfg, name=''):
        self.cfg = cfg
        self.name=name

        from gna.bundle.bundle import get_bundle
        self.bundleclass = get_bundle((cfg.bundle.name, cfg.bundle.get('version', None)))

        variables, objects = self.bundleclass.provides(self.cfg)
        self.items = variables+objects

    def register_in(self, dct):
        if self.cfg.bundle.get('inactive', False):
            return
        for key in self.items:
            dct[key] = self

    def build(self, **kwargs):
        if self.bundle:
            return self.bundle

        self.bundle = self.bundleclass(self.cfg, **kwargs)
        self.bundle.execute()

    def set_nidx(self, nidx):
        if nidx is None:
            printl_debug( 'indices: %s'%(self.name) )
            return

        bundlecfg = self.cfg.bundle
        predefined_nidx = bundlecfg.get('nidx', None)
        if predefined_nidx is None:
            printl_debug( 'indices: %s[%s]'%(self.name, str(predefined_nidx)) )
            bundlecfg.nidx = nidx
        else:
            if isinstance(predefined_nidx, list):
                predefined_nidx = NIndex(fromlist=predefined_nidx)
            elif not isinstance(predefined_nidx, NIndex):
                raise Exception('Unsupported nidx field')

            printl_debug('indices: %s[%s + %s]'%(self.name, str(predefined_nidx), str(nidx)))
            bundlecfg.nidx=predefined_nidx+nidx

class ExpressionContext_v01(object):
    indices = None
    def __init__(self, bundles, ns=None, inputs=None, outputs=None):
        self.bundles = bundles
        self.outputs = NestedDict() if outputs is None else outputs
        self.inputs  = NestedDict() if inputs is None else inputs
        self.ns = ns or env.globalns

        self.providers = dict()
        for name, cfg in self.bundles.items():
            provider = ItemProvider(cfg, name)
            provider.register_in(self.providers)

        self.required_bundles = OrderedDict()

    def __enter__(self):
        self.ns.__enter__()

    def __exit__(self, *args, **kwargs):
        self.ns.__exit__(*args, **kwargs)

    def namespace(self):
        return self.ns

    def set_indices(self, indices):
        self.nindex = indices

    @methodname
    def require(self, name, nidx):
        provider = self.required_bundles.get(name, None)
        if provider is None:
            provider = self.providers.get(name, None)
            if provider is None:
                if nidx:
                    for it in nidx.iterate():
                        self.require(it.current_format(name=name), None)
                    return self.required_bundles

                print('List of available (provided) variables:', list(self.required_bundles.keys()))
                raise Exception('Do not know how to build '+name)

            self.required_bundles[name] = provider

        provider.set_nidx(nidx)

        return self.required_bundles

    def build_bundles(self):
        with self.ns:
            for provider in self.required_bundles.values():
                provider.build(inputs=self.inputs, outputs=self.outputs, namespace=self.ns)

    def get_variable(self, name, *idx):
        pass

    def get_key(self, name, nidx, fmt=None, clone=None):
        if nidx is None:
            nidx = NIndex()
        if clone is not None:
            clone = '%02d'%clone

        if fmt:
            ret = ndix.current_format(fmt)
            if clone:
                ret += '.'+clone
            return ret

        nidx = nidx.current_values(name=name)
        if clone:
            nidx = nidx + (clone,)
        return nidx

    def get_output(self, name, nidx=None, clone=None):
        return self.get( self.outputs, name, nidx, 'output', clone=clone )

    def set_output(self, output, name, nidx=None, fmt=None, **kwargs):
        import ROOT as R
        if isinstance(output, R.TransformationTypes.OutputHandle):
            output = R.OutputDescriptor(output)
        self.set( self.outputs, output, name, nidx, 'output', fmt, **kwargs )
        return output

    def get_input(self, name, nidx=None, clone=None):
        return self.get( self.inputs, name, nidx, 'input', clone=clone )

    def set_input(self, input, name, nidx=None, fmt=None, clone=None):
        self.set( self.inputs, input, name, nidx, 'input', fmt, clone)
        return input

    def get(self, source, name, nidx, type, clone=None):
        key = self.get_key(name, nidx, clone=clone)
        printl_debug('get {}'.format(type), name, key)

        ret = source.get(key, None)
        if not ret:
            raise Exception('Failed to get {} {}[{}]'.format(type, name, nidx, clone))

        if isinstance(ret, NestedDict):
            raise Exception('Incomplete index ({!s}) provided (probably). Need at least resolve {!s}'.format(nidx, res.keys()))

        return ret

    def set(self, target, io, name, nidx, type, fmt=None, clone=None):
        key = self.get_key( name, nidx, fmt, clone )
        printl_debug('set {}'.format(type), name, key)
        target[key]=io

    def set_variable(self, name, nidx, var, **kwargs):
        key = '.'.join(self.get_key( name, nidx ))
        printl_debug('set variable', name, key)
        self.ns.reqparameter(key, cfg=var, **kwargs)

    # def connect(self, source, sink, nidx, fmtsource=None, fmtsink=None):
        # printl_debug( 'connect: {}->{} ({:s})'.format( source, sink, nidx ) )
        # with nextlevel():
            # output = self.get_output( source, nidx )
            # input  = self.get_input( sink, nidx )

        # input( output )

