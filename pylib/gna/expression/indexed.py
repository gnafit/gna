#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.index import *

class Indexed(object):
    name=''
    label=None
    nindex_locked=False
    fmt=None
    expandable=True
    nindex=None
    def __init__(self, name, *indices, **kwargs):
        self.name=name

        indices1=[]
        for idx in indices:
            if isinstance(idx, Indexed):
                idx=idx.nindex
            indices1.append(idx)

        self.set_indices(*indices1, **kwargs)

    def set_label(self, label):
        self.label=label

    def set_format(self, fmt):
        self.fmt = fmt

    def set_indices(self, *indices, **kwargs):
        self.nindex=NIndex(*indices, **kwargs)
        if indices:
            self.nindex_locked=True

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,
        if self.nindex_locked:
            if self.nindex==args:
                return self
            raise Exception('May not modify already declared indices')

        if self.nindex is not None:
            self.set_indices(*args, order=self.nindex.order)
        else:
            self.set_indices(*args)
        return self

    def __add__(self, other):
        raise Exception('not implemented')

    def __str__(self):
        if self.nindex:
            return '{}[{}]'.format(self.name, self.nindex.comma_list())
        else:
            return self.name

    def estr(self, expand=100):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Indexed):
            return False
        if self.name!=other.name:
            return False
        return self.nindex==other.nindex

    def walk(self, yieldself=False, operation=''):
        yield self, operation

    def ident(self, **kwargs):
        if self.name is undefinedname:
            return self.guessname(**kwargs)
        return self.name

    def ident_full(self, **kwargs):
        return '{}:{}'.format(self.ident(**kwargs), self.nindex.ident(**kwargs))

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
        return self.nindex.get_relevant(nidx)

    def current_format(self, nidx, fmt=None, *args, **kwargs):
        nidx = self.nindex.get_relevant(nidx)
        return nidx.current_format( fmt, *args, name=self.name, **kwargs )


