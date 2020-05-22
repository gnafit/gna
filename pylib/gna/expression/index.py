#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import itertools as I
from collections import OrderedDict
from gna.expression.printl import *
from gna.grouping import Groups
import numpy as N

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
    slave, master, group = None, None, None
    def __init__(self, *args, **kwargs):
        first, args = args[0], args[1:]
        if isinstance(first, Index):
            self.short    = first.short
            self.name     = first.name
            self.variants = first.variants
            self.current  = first.current
            if first.slave:
                self.slave    = Index(first.slave, master=self)
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

            if self.variants is not None:
                varcheck = set(self.variants)
                if len(varcheck)!=len(self.variants):
                    raise Exception('There are duplicated indices ({name}): {idxlist!s}'.format(name=self.name, idxlist=self.variants))

            slave = kwargs.pop('slave', None)
            if slave:
                self.set_slave(slave)

        self.master=kwargs.pop('master', None)

        if args or kwargs:
            raise Exception( 'Unparsed paramters: {:s}, {:s}'.format(args, kwargs) )

    def get_size(self):
        return len(self.variants)

    def set_slave(self, slave):
        short = slave['short']
        name = slave['name']
        map = slave['map']
        if isinstance(map, (list, tuple)):
            map=OrderedDict(map)
        variants = list(map.keys())

        self.slave = Index(short, name, variants, master=self)
        self.group = Groups(map)

        for v in self.variants:
            if not v in self.group:
                raise Exception('Inconsistent slave index {}'.format(v))

        if self.current:
            self.slave.current=self.group[self.current]

    def set_current(self, c):
        self.current = c
        if self.slave:
            self.slave.current=self.group[self.current]

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

    def current_items(self, mode='short', include_slaves=False):
        res=tuple()
        if mode in ('short', 'both'):
            res+=(self.short, self.current),
        if mode in ('long', 'both'):
            res+=(self.name, self.current),

        if not res:
            raise Exception( 'Unknown mode '+mode )

        if include_slaves and self.slave:
            res+=self.slave.current_items(mode)

        return res

    def current_values(self, include_slaves=False):
        res=(self.current,)

        if include_slaves and self.slave:
            res+=self.slave.current_values(include_slaves)

        return res

    def configure_override(self, slaveof, masterof):
        if not self.slave:
            return

        slaveof[self.short]=self.slave
        masterof[self.slave.short]=self

class NIndex(object):
    name_position=0
    def __init__(self, *indices, **kwargs):
        self.indices = OrderedDict()

        self.slaveof={}
        self.masterof={}

        self.order_initial=[]
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
        name_position=0
        for i, args in enumerate(fromlist):
            if isinstance(args, str) and args=='name':
                name_position=i
                continue
            args, slave=args[:3], args[3:]
            slave=slave and slave[0] or None
            idx = Index(*args, slave=slave)
            self._append_indices(idx)

        self.arrange(kwargs.pop('order', self.order), kwargs.pop('name_position', name_position))

        if kwargs:
            raise Exception('Unparsed kwargs: {:s}'.format(kwargs))

    @classmethod
    def fromlist(cls, lst):
        return cls(fromlist=lst)

    def __str__(self):
        s = ['%s=%s'%(v.short, v.current) if v.current is not None else v.short for v in self.indices.values()]
        return 'NIndex(%s)'%( ', '.join(s) )

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
        from gna.expression.indexed import Indexed
        if isinstance(other, Index):
            self._set_new(other.short, other)
        elif isinstance(other, str):
            self._set_new(other, Index(other, other, variants=None))
        else:
            neworder = None
            if isinstance(other, NIndex):
                others = other.indices.values()
                neworder=other.order
            elif isinstance(other, Indexed):
                others = other.nindex.indices.values()
                neworder=other.nindex.order
            else:
                raise Exception( 'Unsupported index type '+type(other).__name__ )

            self.orders_consistent(self.order, neworder, True)
            self.order=neworder

            for other in others:
                self._set_new(other.short, other)

        return self

    def _set_new(self, short, other):
        self.order_initial.append(short)
        if short in self.masterof:
            return

        self.indices[short]=other
        other.configure_override(slaveof=self.slaveof, masterof=self.masterof)

        if short in self.slaveof:
            slave=self.slaveof[short]
            if slave.short in self.indices:
                del self.indices[slave.short]

    def make_inheritor(self, *args, **kwargs):
        kwargs.setdefault('order', self.order)
        return NIndex(*args, **kwargs)

    def __sub__(self, other):
        return self.make_inheritor(self, ignore=other.indices.keys())

    def arrange(self, order, name_position=0):
        if order:
            if order=='sorted':
                self.order = sorted(self.order_initial)
                self.order.insert(name_position, 'name')
            else:
                self.order = order
        else:
            self.order = self.order_initial
            if not 'name' in self.order:
                self.order.insert(name_position, 'name')

        self.order_indices=list((name for name in self.order if name in self.indices))

        self.indices = OrderedDict([(k, self.indices[k]) for k in self.order_indices if k in self.indices])

    # def __str__(self):
        # return ', '.join( self.indices.keys() )

    def __bool__(self):
        return bool(self.indices)

    __nonzero__ = __bool__

    def __eq__(self, other):
        if not isinstance(other, NIndex):
            other = NIndex(*other)
        return self.indices==other.indices

    def ident(self, **kwargs):
        return '_'.join(self.indices.keys())

    def comma_list(self, **kwargs):
        return ','.join(self.indices.keys())

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

    def current_values(self, name=None, include_slaves=False):
        ret = tuple()
        for short in self.order:
            idx = self.indices.get(short, None)
            if idx is not None:
                ret+=(idx.current,)
                continue

            if include_slaves:
                idx = self.masterof.get(short, None)
                if idx:
                    ret+=(idx.current,)
                    continue

            if name is not None and short=='name':
                ret+=(name,)

        return ret

    def current_items(self, mode='short', include_slaves=False):
        ret = tuple()
        for short in self.order:
            idx = self.indices.get(short, None)
            if idx is not None:
                ret+=idx.current_items(mode)

            if include_slaves:
                idx = self.slaveof.get(short, None)
                if idx:
                    ret+=idx.current_items(mode)

        return ret

    def make_format_string(self, with_name):
        names=self.names(False, with_name)
        autofmt = '}.{'.join(names)
        return autofmt and '{%s}'%autofmt or ''

    def current_format(self, fmt=None, *args, **kwargs):
        fmtauto = self.make_format_string(False)

        dct = dict( self.current_items('both', include_slaves=True)+args, **kwargs )
        indexauto = fmtauto.format(**dct)

        if not fmt and 'name' in kwargs:
            fmt = self.make_format_string(True)

        if fmt:
            dct['autoindex'] = indexauto
        else:
            return indexauto

        return fmt.format( **dct )

    def get_relevant_index(self, short, exception=True):
        idx = self.indices.get(short, None)
        if idx is not None:
            return idx

        master = self.masterof.get(short, None)
        if master is None:
            if exception:
                raise Exception('Can not find relevant index for {} in {}'.format(short, self.indices.keys()))
            else:
                return None

        return master.slave

    def get_relevant(self, indices):
        if isinstance(indices, NIndex):
            return self.make_inheritor(*[indices.get_relevant_index(s) for s in self.indices.keys()])

        lst=()
        for s in indices:
            idx=self.get_relevant_index(s, exception=False)
            if idx:
                lst+=idx,
        return self.make_inheritor(*lst)

    def get_subset(self, indices):
        if isinstance(indices, NIndex):
            return self.make_inheritor(*[self.get_relevant_index(s) for s in indices.indices.keys()])

        return self.make_inheritor(*[self.get_relevant_index(s) for s in indices])

    def split(self, indices):
        majors, minors, used=(), (), ()

        for short in indices:
            major=self.get_relevant_index(short)
            majors+=major,
            used+=short,
            if major.master:
                used+=major.master.short,

        for short, idx in self.indices.items():
            if short in used:
                continue

            minors+=idx,

        return self.make_inheritor(*majors), self.make_inheritor(*minors)

    def get_current(self, short):
        return self.indices[short].current

    def get_index_names(self):
        return tuple(idx.name for idx in self.indices.values())

    def ndim(self):
        return len(self.indices)

    def get_size(self):
        return N.product([idx.get_size() for idx in self.indices.values()])

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
