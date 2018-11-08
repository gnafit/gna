# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import NestedDict
from gna.expression.preparse import open_fcn
from gna.expression.operation import *
from gna.env import env
import re

class VTContainer(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(VTContainer, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        newvar = Variable(key)
        self[key] = newvar
        return newvar

    def __setitem__(self, key, value):
        if isinstance(value, Indexed):
            if value.name is undefinedname and key!='__tree__':
                value.name = key

        OrderedDict.__setitem__(self, key, value)
        return value

class Expression(object):
    operations = dict(sum=OSum, prod=OProd, concat=OConcat, accumulate=Accumulate, bracket=bracket)
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

        self.globals=VTContainer(self.operations)
        self.indices=OrderedDict()
        self.defindices(indices, **kwargs)

    def parse(self):
        if self.tree:
            raise Exception('Expression is already parsed')

        self.trees = []
        for expr in self.expressions:
            texpr = '__tree__ = '+expr
            try:
                exec(texpr, self.globals)
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
            lib[v['expr']] = v
        for tree in self.trees:
            tree.guessname(lib, *args, **kwargs)

    def __str__(self):
        return self.expressions_raw

    def __repr__(self):
        return 'Expression("{}")'.format(self.expressions_raw)

    def defindices(self, defs):
        self.indices = NIndex(fromlist=defs)
        for short, idx in self.indices.indices.items():
            self.globals[short] = idx

            sub=idx.sub
            if sub:
                self.globals[sub.short]=sub

    def build(self, context):
        if not self.tree:
            raise Exception('Expression is not initialized, call parse() method first')

        context.set_indices(self.indices)
        for tree in self.trees:
            creq = tree.require(context)

        context.build_bundles()

        for tree in self.trees:
            tree.bind(context)

class ExpressionContext(object):
    indices = None
    executed_bundes = set()
    required = OrderedDict()
    def __init__(self, cfg, ns=None):
        self.cfg = cfg
        self.outputs = NestedDict()
        self.inputs  = NestedDict()
        self.ns = ns or env.globalns

        self.providers = dict()
        for keys, value in cfg.items():
            if isinstance(value, NestedDict) and 'provides' in value:
                value.provides+=[keys]
                keys=value.provides

            if not isinstance(keys, (list, tuple)):
                keys=keys,

            for key in keys:
                self.providers[key]=value

    def namespace(self):
        return self.ns

    def set_indices(self, indices):
        self.indices = indices

    @methodname
    def require(self, name, indices):
        cfg = self.required.get(name, None)
        if cfg is None:
            cfg = self.providers.get(name, None)
            if cfg is None:
                if indices:
                    fmt='{name}{autoindex}'
                    for it in indices.iterate():
                        self.require(it.current_format(fmt, name=name), None)
                    return self.required

                raise Exception('Do not know how to build '+name)

            self.required[name] = cfg

        if indices is None:
            printl_debug( 'indices: %s'%(name) )
            return self.required

        predefined = cfg.get('indices', None)
        if predefined is None:
            printl_debug( 'indices: %s[%s]'%(name, str(indices)) )
            cfg.indices=indices
        elif not isinstance(predefined, NIndex):
            raise Exception('Configuration should not contain predefined "indices" field')
        else:
            printl_debug( 'indices: %s[%s + %s]'%(name, str(predefined), str(indices)) )
            cfg.indices=predefined+indices

        return self.required

    def build_bundles(self):
        done = set()
        for cfg in self.required.values():
            if cfg in done:
                continue
            self.build_bundle(cfg)
            done.add(cfg)

    def build_bundle(self, cfg):
        printl_debug('build bundle', cfg.bundle )

        from gna.bundle import execute_bundle
        with nextlevel():
            b=execute_bundle( cfg=cfg, context=self )

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

        nidx = nidx.current_values()
        if clone:
            nidx = nidx + (clone,)
        return (name,)+nidx

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
