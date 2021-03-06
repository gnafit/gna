#!/usr/bin/env python

from gna.expression.indexed import *

def call_once(method):
    def newmethod(self, *args, **kwargs):
        if not hasattr(self, 'call_once'):
            self.call_once=set()

        if method in self.call_once:
            return

        method(self, *args, **kwargs)

        self.call_once.add(method)

    return newmethod

class Variable(Indexed):
    callable = None
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, Transformation):
            from gna.expression.compound import WeightedTransformation
            return WeightedTransformation(undefinedname, self, other)

        from gna.expression.compound import VProduct
        return VProduct(undefinedname, self, other)

    def __call__(self, *targs):
        if self.callable:
            return self.callable

        from gna.expression.compound import TCall
        self.callable=TCall(self.name, self, targs=targs)
        return self.callable

    @call_once
    def bind(self, context):
        pass

    def get_output(self, nidx, context):
        pass

    @methodname
    def require(self, context):
        context.require(self.name, self.nindex)

class Transformation(Indexed):
    def __init__(self, name, *args, **kwargs):
        Indexed.__init__(self, name, *args, **kwargs)

    def __str__(self):
        return '{}()'.format(Indexed.__str__(self))

    def __div__(self, other):
        from gna.expression.compound import TRatio
        return TRatio(undefinedname, self, other)

    def __floordiv__(self, other):
        from gna.expression.compound import TRatio
        return TRatio(undefinedname, self, other, broadcast=True)

    def __truediv__(self, other):
        from gna.expression.compound import TRatio
        return TRatio(undefinedname, self, other)

    def __mul__(self, other, may_weight=True):
        from gna.expression.compound import WeightedTransformation
        if may_weight:
            if isinstance(other, Variable):
                return WeightedTransformation(undefinedname, self, other)
            elif isinstance(self, WeightedTransformation) and self.expandable and other.expandable:
                return WeightedTransformation(undefinedname, self, other)
            elif isinstance(other, WeightedTransformation) and self.expandable and other.expandable:
                return WeightedTransformation(undefinedname, self, other)

        from gna.expression.compound import TProduct
        return TProduct(undefinedname, self, other)

    def __add__(self, other):
        from gna.expression.compound import TSum
        return TSum(undefinedname, self, other)

    @methodname
    def require(self, context):
        context.require(self.name, self.nindex)

    @call_once
    def bind(self, context):
        pass
