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

    def bind(self, context):
        pass

    def get_output(self, nidx, context):
        pass

    def __getattr__(self, name):
        self.name = '.'.join(self.name, name)
        return self

    @methodname
    def require(self, context):
        context.require(self.name, self.indices)

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

    @methodname
    def require(self, context):
        context.require(self.name, self.indices)

    def bind(self, context):
        pass
