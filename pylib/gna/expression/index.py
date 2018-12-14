#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import itertools as I
from collections import OrderedDict
from gna.expression.printl import *
from gna.grouping import Groups

debugmethods=False
if debugmethods:
    def methodname(fcn):
        def newfcn(self, *args, **kwargs):
            printl('methodname', type(self).__name__, getattr(self, 'name', '?'), fcn.__name__, *args, **kwargs)
            with nextlevel():
                return fcn(self, *args, **kwargs)
        return newfcn
    printl_debug = printl
else:
    def methodname(fcn):
        return fcn
    def printl_debug(*args, **kwargs):
        pass

class Index(object):
    sub, group = None, None
    def __init__(self, *args, **kwargs):
        first, args = args[0], args[1:]
        if isinstance(first, Index):
            self.short    = first.short
            self.name     = first.name
            self.variants = first.variants
            self.current  = first.current
            if first.sub:
                self.sub      = Index(first.sub)
                self.group    = first.group
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

            sub = kwargs.pop('sub', None)
            if sub:
                self.set_sub(sub)

        if args or kwargs:
            raise Exception( 'Unparsed paramters: {:s}, {:s}'.format(args, kwargs) )

    def set_sub(self, sub):
        short = sub['short']
        name = sub['name']
        map = sub['map']
        variants = list(map.keys())

        self.sub = Index(short, name, variants)
        self.group = Groups(map)

        for v in self.variants:
            if not v in self.group:
                raise Exception('Inconsistent sub index {}'.format(v))

        if self.current:
            self.sub.current=self.group[self.current]

    def set_current(self, c):
        self.current = c
        if self.sub:
            self.sub.current=self.group[self.current]

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
            ret = Index(self)
            ret.set_current(var)
            yield ret

    __iter__ = iterate

    def __str__(self):
        return '{name} ({short}): {variants:s}'.format( **self.__dict__ )

    def current_items(self, mode='short', include_sub=False):
        res=tuple()
        if mode in ('short', 'both'):
            res+=(self.short, self.current),
        if mode in ('long', 'both'):
            res+=(self.name, self.current),

        if not res:
            raise Exception( 'Unknown mode '+mode )

        if include_sub and self.sub:
            res+=self.sub.current_items(mode)

        return res

    def configure_override(self, overridden, overriding):
        if not self.sub:
            return

        main, sub = self.short, self.sub.short
        overridden[sub]=main
        overriding[main]=sub

class NIndex(object):
    name_position=0
    def __init__(self, *indices, **kwargs):
        self.indices = OrderedDict()
        self.overriding={}
        self.overridden={}
        self.order=[]
        self.order_indices=[]

        for idx in indices:
            self._append_indices(idx)

        ignore = kwargs.pop('ignore', None)
        if ignore:
            for other in ignore:
                if other in self.indices:
                    del self.indices[other]

        fromlist = kwargs.pop('fromlist', [])
        for args in fromlist:
            args, sub=args[:3], args[3:]
            sub=sub and sub[0] or None
            idx = Index(*args, sub=sub)
            self._append_indices(idx)

        self.arrange(kwargs.pop('order', None), kwargs.pop('name_position', 0))

        if kwargs:
            raise Exception('Unparsed kwargs: {:s}'.format(kwargs))

    def orders_consistent(self, order1, order2, exception=False):
        if not order1 or order1==['name']:
            return True
        if not order2 or order2==['name']:
            return True

        ret=order1==order2

        if exception and not ret:
            print('Order 1', order1)
            print('Order 2', order2)
            raise Exception('Index orders are inconsistent')

        return ret

    def __add__(self, other):
        if not isinstance(other, NIndex):
            raise Exception('Unsupported add() type')

        self.orders_consistent(self.order, other.order, True)

        return NIndex(self, other, order=self.order)

    def _append_indices(self, other):
        if isinstance(other, Index):
            self.set_new(other.short, other)
        elif isinstance(other, str):
            self.set_new(other, Index(other, other, variants=None))
        else:
            neworder = None
            if isinstance(other, NIndex):
                others = other.indices.values()
                neworder=other.order
            elif isinstance(other, Indexed):
                others = other.indices.indices.values()
                neworder=other.indices.order
            else:
                raise Exception( 'Unsupported index type '+type(other).__name__ )

            self.orders_consistent(self.order, neworder, True)
            self.order=neworder

            for other in others:
                self.set_new(other.short, other)

        return self

    def set_new(self, short, other):
        if short in self.overridden:
            return

        self.indices[short]=other
        other.configure_override( overriding=self.overriding, overridden=self.overridden )

        if short in self.overriding:
            oshort=self.overriding[short]
            if oshort in self.indices:
                del self.indices[oshort]

    def make_inheritor(self, *args, **kwargs):
        kwargs.setdefault('order', self.order)
        return NIndex(*args, **kwargs)

    def __sub__(self, other):
        return self.make_inheritor(self, ignore=other.indices.keys())

    def arrange(self, order, name_position=0):
        if order:
            if order=='sorted':
                self.order = sorted(self.indices.keys())
                self.order.insert(name_position, 'name')
            else:
                self.order = order
        else:
            self.order = self.indices.keys()
            self.order.insert(name_position, 'name')

        self.order_indices=list(self.order)
        self.order_indices.remove('name')

        self.indices = OrderedDict([(k, self.indices[k]) for k in self.order_indices if k in self.indices])

    def __str__(self):
        return ', '.join( self.indices.keys() )

    def __bool__(self):
        return bool(self.indices)

    __nonzero__ = __bool__

    def __eq__(self, other):
        if not isinstance(other, NIndex):
            other = NIndex(*other)
        return self.indices==other.indices

    def ident(self, **kwargs):
        return '_'.join(self.indices.keys())

    def names(self, short=False, with_name=False):
        ret = []
        for k in self.order:
            idx = self.indices.get(k, None)
            if idx is not None:
                ret.append(short and idx.short or idx.name)
            elif with_name and k=='name':
                ret.append(k)
        return ret

    def iterate(self, fix={}, **kwargs):
        for it in I.product(*(idx.iterate(fix=fix) for idx in self.indices.values())):
            yield self.make_inheritor(*(Index(idx) for idx in it))

    __iter__ = iterate

    def current_values(self):
        return tuple(idx.current for idx in self.indices.values())

    def current_items(self, mode='short', *args, **kwargs):
        res=()
        include_sub=kwargs.pop('include_sub', None)
        for v in self.indices.values():
            res+=v.current_items(mode, *args, include_sub=include_sub)
        if mode=='both':
            res+=args+tuple(kwargs.items())
        return res

    def make_format_string(self, with_name):
        names=self.names(False, with_name)
        autofmt = '}.{'.join(names)
        return autofmt and '{%s}'%autofmt or ''

    def current_format(self, fmt=None, *args, **kwargs):
        fmtauto = self.make_format_string(False)

        dct = dict( self.current_items('both', include_sub=True)+args, **kwargs )
        indexauto = fmtauto.format(**dct)

        if not fmt and 'name' in kwargs:
            fmt = self.make_format_string(True)

        if fmt:
            dct['autoindex'] = indexauto
        else:
            return indexauto

        return fmt.format( **dct )

    def get_relevant_index(self, short):
        idx = self.indices.get(short, None)
        if idx is not None:
            return idx

        oshort = self.overridden.get(short, None)
        if oshort is None:
            raise Exception('Can not find relevant index for {} in {}.format(short, self.indices.keys())')

        return self.indices[oshort].sub

    def get_relevant(self, nidx):
        return self.make_inheritor(*[nidx.get_relevant_index(s) for s in self.indices.keys()])

    def get_sub(self, indices):
        sub_indices=[v for k, v in self.indices.items() if k in indices]
        return self.make_inheritor(*sub_indices)

    def split(self, indices):
        idx=[]
        other=[]
        for (k, v) in self.indices.items():
            if k in indices:
                idx.append(v)
            else:
                other.append(v)

        return self.make_inheritor(*idx), self.make_inheritor(*other)

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

    def __repr__(self):
        return 'NameUndefined()'

    def __add__(self, other):
        return self.__str__()+other

undefinedname = NameUndefined()

class Indexed(object):
    name=''
    label=None
    indices_locked=False
    fmt=None
    expandable=True
    indices=None
    def __init__(self, name, *indices, **kwargs):
        self.name=name

        indices1=[]
        for idx in indices:
            if isinstance(idx, Indexed):
                idx=idx.indices
            indices1.append(idx)

        self.set_indices(*indices1, **kwargs)

        print('Indices', self.name, self.indices.order, self.indices.indices.keys())

    def set_label(self, label):
        self.label=label

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
            printl(operator, obj, prefix=('% 3i'%i, '% 2i'%current_level()) )

    def get_output(self, nidx, context):
        return context.get_output(self.name, self.get_relevant( nidx ))

    def get_input(self, nidx, context, clone=None):
        return context.get_input(self.name, self.get_relevant( nidx ), clone=clone)

    def get_relevant(self, nidx):
        return self.indices.get_relevant(nidx)

    def current_format(self, nidx, fmt=None, *args, **kwargs):
        nidx = self.indices.get_relevant(nidx)
        return nidx.current_format( fmt, *args, name=self.name, **kwargs )

