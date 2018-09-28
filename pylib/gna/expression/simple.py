#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.index import *

class Variable(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, Transformation):
            from gna.expression.compound import WeightedTransformation
            return WeightedTransformation(undefinedname, self, other)

        from gna.expression.compound import VProduct
        return VProduct(undefinedname, self, other)

    def __call__(self, *targs):
        from gna.expression.compound import TCall
        return TCall(self.name, self, targs=targs)

    def build(self, context):
        printl('build (var) {}:'.format(type(self).__name__), str(self) )
        with nextlevel():
            context.build(self.name, self.indices)

    def get_output(self, nidx, context):
        pass

    def __getattr__(self, name):
        self.name = '.'.join(self.name, name)
        return self

class Transformation(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Transformation, self).__init__(name, *args, **kwargs)

    def __str__(self):
        return '{}()'.format(Indexed.__str__(self))

    def __mul__(self, other):
        from gna.expression.compound import WeightedTransformation
        if isinstance(other, (Variable, WeightedTransformation)):
            return WeightedTransformation(undefinedname, self, other)

        from gna.expression.compound import TProduct
        return TProduct(undefinedname, self, other)

    def __add__(self, other):
        from gna.expression.compound import TSum
        return TSum(undefinedname, self, other)

    def build(self, context):
        printl('build (trans) {}:'.format(type(self).__name__), str(self) )
        context.build(self.name, self.indices)
