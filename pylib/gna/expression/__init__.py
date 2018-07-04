# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import NestedDict
from gna.expression.preparse import open_fcn
from gna.expression.operation import *
from gna.env import env

class VTContainer(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(VTContainer, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        newvar = Variable(key)
        self[key] = newvar
        return newvar

class Expression(object):
    operations = dict(sum=OSum, prod=OProd)
    tree = None
    def __init__(self, expression, indices=[], **kwargs):
        self.expression_raw = expression
        self.expression = open_fcn( self.expression_raw )

        self.globals=VTContainer(self.operations)
        self.indices=OrderedDict()
        self.defindices(indices, **kwargs)

    def parse(self):
        if self.tree:
            raise Exception('Expression is already parsed')
        self.tree=eval(self.expression, self.globals)

    def guessname(self, ilib, *args, **kwargs):
        lib = dict()
        for k, v in ilib.items():
            v['name'] = k
            lib[v['expr']] = v
        self.tree.guessname(lib, *args, **kwargs)

    def __str__(self):
        return self.expression_raw

    def __repr__(self):
        return 'Expression("{}")'.format(self.expression_raw)

    def defindices(self, defs):
        self.indices = NIndex(fromlist=defs)
        for short, idx in self.indices.indices.items():
            self.globals[short] = idx

    def build(self, context):
        if not self.tree:
            raise Exception('Expression is not initialized, call parse() method first')

        context.set_indices(self.indices)
        self.tree.build( context )

class ExpressionContext(object):
    indices = None
    def __init__(self, cfg, ns=None):
        self.cfg = cfg
        self.outputs = NestedDict()
        self.inputs  = NestedDict()
        self.ns = ns or env.globalns

        self.providers = dict()
        for keys, value in cfg.items():
            if isinstance(value, NestedDict) and 'provides' in value:
                keys=value.provides+[keys]

            if not isinstance(keys, (list, tuple)):
                keys=keys,

            for key in keys:
                self.providers[key]=value

    def set_indices(self, indices):
        self.indices = indices

    def build(self, name, indices):
        cfg = self.providers.get(name, None)
        if cfg is None:
            if indices:
                fmt='{name}{autoindex}'
                for it in indices.iterate():
                    self.build(it.current_format(fmt, name=name), None)
                return

            raise Exception('Do not know how to build '+name)

        printl('build', name, 'via bundle' )

        if indices is not None:
            cfg.indices=indices

        from gna.bundle import execute_bundle
        with nextlevel():
            b=execute_bundle( cfg=cfg, context=self )

    def get_variable(self, name, *idx):
        pass

    def check_outputs(self, obj):
        printl( 'check', obj )
        with nextlevel():
            if obj.name in self.outputs or obj.name in self.ns.storage or obj.name in self.ns.namespaces:
                printl( 'found' )
                return

            obj.build(self)

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

    def get_output(self, name, nidx, clone=None):
        return self.get( self.outputs, name, nidx, 'output', clone=clone )

    def set_output(self, output, name, nidx=None, fmt=None, **kwargs):
        self.set( self.outputs, output, name, nidx, 'output', fmt, **kwargs )

    def get_input(self, name, nidx=None, clone=None):
        return self.get( self.inputs, name, nidx, 'input', clone=clone )

    def set_input(self, input, name, nidx, fmt=None, clone=None):
        self.set( self.inputs, input, name, nidx, 'input', fmt, clone)

    def get(self, source, name, nidx, type, clone=None):
        key = self.get_key(name, nidx, clone=clone)
        printl('get {}'.format(type), name, key)

        ret = source.get(key, None)
        if not ret:
            raise Exception('Failed to get {} {}[{}]'.format(type, name, nidx, clone))

        return ret

    def set(self, target, io, name, nidx, type, fmt=None, clone=None):
        key = self.get_key( name, nidx, fmt, clone )
        printl('set {}'.format(type), name, key)
        target[key]=io

    def set_variable(self, name, nidx, var, **kwargs):
        key = '.'.join(self.get_key( name, nidx ))
        printl('set variable', name, key)
        self.ns.reqparameter(key, cfg=var, **kwargs)

    # def connect(self, source, sink, nidx, fmtsource=None, fmtsink=None):
        # printl( 'connect: {}->{} ({:s})'.format( source, sink, nidx ) )
        # with nextlevel():
            # output = self.get_output( source, nidx )
            # input  = self.get_input( sink, nidx )

        # input( output )
