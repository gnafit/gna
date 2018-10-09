#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import itertools as I
from collections import OrderedDict
from gna.expression.printl import *

class Index(object):
    def __init__(self, *args, **kwargs):
        first, args = args[0], args[1:]
        if isinstance(first, Index):
            self.short    = first.short
            self.name     = first.name
            self.variants = first.variants
            self.current  = first.current
        else:
            self.short = first
            if len(args)>1:
                self.name, self.variants = args
            else:
                self.name=self.short
                self.variants=kwargs.pop('variants', [self.name])
            args = []
            self.current=kwargs.pop( 'current', None )

        varcheck = set(self.variants)
        if len(varcheck)!=len(self.variants):
            raise Exception('There are duplicated indices ({name}): {idxlist!s}'.format(name=self.name, idxlist=self.variants))

        if args or kwargs:
            raise Exception( 'Unparsed paramters: {:s}, {:s}'.format(args, kwargs) )

    def iterate(self, fix={}):
        if self.variants is None:
            raise Exception( 'Variants are not initialized for {name}'.format(**self.__dict__) )

        val = fix.get(self.name, fix.get(self.short, None))
        if val is not None:
            if not val in self.variants:
                raise Exception( 'Can not fix index {name} in value {value}. Variants are: {variants:s}'.format( **self.__dict__ ) )
            variants = val,
        else:
            variants = self.variants

        for var in variants:
            ret = Index(self.short, self.name, self.variants, current=var)
            yield ret

    __iter__ = iterate

    def __str__(self):
        return '{name} ({short}): {variants:s}'.format( **self.__dict__ )

class NIndex(object):
    def __init__(self, *indices, **kwargs):
        self.indices = OrderedDict()

        for idx in indices:
            self |= idx

        ignore = kwargs.pop('ignore', None)
        if ignore:
            for other in ignore:
                if other in self.indices:
                    del self.indices[other]

        fromlist = kwargs.pop('fromlist', [])
        for args in fromlist:
            idx = Index(*args)
            self |= idx

        self.arrange()

        if kwargs:
            raise Exception('Unparsed kwargs: {:s}'.format(kwargs))

    def __add__(self, other):
        if not isinstance(other, NIndex):
            raise Exception('Unsupported add() type')

        return NIndex(self, other)

    def __ior__(self, other):
        if isinstance(other, Index):
            self.indices[other.short]=other
        elif isinstance(other, str):
            self.indices[other]=Index(other, other, variants=None)
        else:
            if isinstance(other, NIndex):
                others = other.indices.values()
            elif isinstance(other, Indexed):
                others = other.indices.indices.values()
            else:
                raise Exception( 'Unsupported index type '+type(other).__name__ )

            for other in others:
                self.indices[other.short]=other

        return self

    def __sub__(self, other):
        return NIndex(self, ignore=other.indices.keys())

    def arrange(self):
        self.indices = OrderedDict( [(k, self.indices[k]) for k in sorted(self.indices.keys())] )

    def __str__(self):
        return ', '.join( self.indices.keys() )

    def __add__(self, other):
        return NIndex(self, other)

    def __bool__(self):
        return bool(self.indices)

    __nonzero__ = __bool__

    def __eq__(self, other):
        if not isinstance(other, NIndex):
            other = NIndex(*other)
        return self.indices==other.indices

    # def reduce(self, *indices):
        # if not set(indices.keys()).issubset(self.indices.keys()):
            # raise Exception( "NIndex.reduce should be called on a subset of indices, got {:s} in {:s}".format(indices.keys(), self.indices.keys()) )

        # return NIndex(*(set(self.indices)-set(indices))) #legacy

    def ident(self, **kwargs):
        return '_'.join(self.indices.keys())

    def names(self, short=False):
        if short:
            return [idx.short for idx in self.indices.values()]
        else:
            return [idx.name for idx in self.indices.values()]

    def iterate(self, fix={}, **kwargs):
        for it in I.product(*(idx.iterate(fix=fix) for idx in self.indices.values())):
            yield NIndex(*(Index(idx) for idx in it))

    __iter__ = iterate

    def current_values(self):
        return tuple(idx.current for idx in self.indices.values())

    def current_items(self, mode='short', *args, **kwargs):
        if mode=='short':
            return tuple((idx.short, idx.current) for idx in self.indices.values())
        elif mode=='long':
            return tuple((idx.name, idx.current) for idx in self.indices.values())
        elif mode=='both':
            return tuple((idx.short, idx.current) for idx in self.indices.values())+tuple((idx.name, idx.current) for idx in self.indices.values())+args+tuple(kwargs.items())

        raise Exception( 'Unknown mode '+mode )

    def autofmt(self):
        autofmt = '}.{'.join(self.names())
        if autofmt:
            return '{'+autofmt+'}', '.{'+autofmt+'}'
        return autofmt, autofmt

    def current_format(self, fmt=None, *args, **kwargs):
        autofmtnd, autofmt = self.autofmt()
        dct = dict( self.current_items('both')+args, **kwargs )
        autoindexnd, autoindex = autofmtnd.format(**dct), autofmt.format(**dct)
        if fmt:
            dct['autoindex'] = autoindex
            dct['autoindexnd'] = autoindexnd
        else:
            return autoindex
        return fmt.format( **dct )

    def get_relevant(self, nidx):
        return NIndex(*[v for k, v in nidx.indices.items() if k in self.indices])

    def get_sub(self, indices):
        return NIndex(*[v for k, v in self.indices.items() if k in indices])

    def split(self, indices):
        idx=[]
        other=[]
        for k, v in self.indices.items():
            (idx if k in indices else other).append(v)

        return NIndex(*idx), NIndex(*other)

    def get_current(self, short):
        return self.indices[short].current

    def ndim(self):
        return len(self.indices)

    def __contains__(self, other):
        for name in other.indices.keys():
            if not name in self.indices:
                return False

        return True

class NameUndefined():
    def __str__(self):
        return '?'

    def __add__(self, other):
        return self.__str__()+other

undefinedname = NameUndefined()

class Indexed(object):
    name=''
    indices_locked=False
    fmt=None
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        self.set_indices(*indices, **kwargs)

    def set_format(self, fmt):
        self.fmt = fmt

    def set_indices(self, *indices, **kwargs):
        self.indices=NIndex(*indices, **kwargs)
        if indices:
            self.indices_locked=True

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,
        if self.indices_locked:
            if self.indices==args:
                return self
            raise Exception('May not modify already declared indices')
        self.set_indices(*args)
        return self

    def __add__(self, other):
        raise Exception('not implemented')

    def __str__(self):
        if self.indices:
            return '{}[{}]'.format(self.name, str(self.indices))
        else:
            return self.name

    def estr(self, expand=100):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Indexed):
            return False
        if self.name!=other.name:
            return False
        return self.indices==other.indices

    # def reduce(self, newname, *indices):
        # return Indexed(newname, self.indices.reduce(*indices))

    def walk(self, yieldself=False, operation=''):
        yield self, operation

    def ident(self, **kwargs):
        if self.name is undefinedname:
            return self.guessname(**kwargs)
        return self.name

    def ident_full(self, **kwargs):
        return '{}:{}'.format(self.ident(**kwargs), self.indices.ident(**kwargs))

    def guessname(self, *args, **kwargs):
        return undefinedname

    def dump(self, yieldself=False):
        for i, (obj, operator) in enumerate(self.walk(yieldself)):
            printl(operator, obj, prefix=(i, printlevel) )

    def get_output(self, nidx, context):
        return context.get_output(self.name, self.get_relevant( nidx ))

    def get_input(self, nidx, context, clone=None):
        return context.get_input(self.name, self.get_relevant( nidx ), clone=clone)

    def get_relevant(self, nidx):
        return self.indices.get_relevant(nidx)

    def current_format(self, nidx, fmt=None, *args, **kwargs):
        nidx = self.indices.get_relevant(nidx)
        if not fmt:
            fmt = '{name}{autoindex}'
        return nidx.current_format( fmt, *args, name=self.name, **kwargs )

