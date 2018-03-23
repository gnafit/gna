# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict

class Indices(object):
    def __init__(self, *indices, **kwargs):
        self.indices = set()

        for idx in indices:
            if isinstance(idx, Indices):
                self.indices |= set(idx.indices)
            elif isinstance(idx, str):
                self.indices.add(idx)
            else:
                raise Exception( 'Unsupported index type '+type(idx).__name__ )

        self.indices=list(sorted(self.indices))

        if kwargs:
            raise Exception('Unparsed kwargs')

    def __str__(self):
        return ', '.join( self.indices )

    def __add__(self, other):
        return Indices(self, other)

    def __bool__(self):
        return bool(self.indices)

    __nonzero__=__bool__

    def __eq__(self, other):
        return self.indices==other.indices

    def reduce(self, *indices):
        if not set(indices).issubset(self.indices):
            raise Exception( "Indices.reduce should be called on a subset of indices" )

        return Indices(*(set(self.indices)-set(indices)))


class Indexed(Indices):
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        super(Indexed,self).__init__(*indices, **kwargs)

    def __str__(self):
        if self.indices:
            return '{}[{}]'.format(self.name, Indices.__str__(self))
        else:
            return self.name

    def __eq__(self, other):
        if self.name!=other.name:
            return False
        return self.indices==other.indices

    def reduce(self, newname, *indices):
        return Indexed( newname, Indices.reduce(self, *indices) )

class Variable(Indexed):
    def __init__(self, name, *indices, **kwargs):
        super(Variable, self).__init__(name, *indices, **kwargs)

    def __mul__(self, other):
        return VProduct(self, other)

    def __call__(self):
        return Transformation(self.name)

class VProduct(Variable):
    def __init__(self, *objects, **kwargs):
        name = kwargs.pop('name', '')
        for o in objects:
            if not isinstance(o, Variable):
                raise Exception('Expect Variable instance')

        self.objects=list(objects)
        super(VProduct, self).__init__(name, *objects)

    def estr(self):
        return '{}'.format( ' * '.join(str(o) for o in self.objects) )

    def __mul__(self, other):
        if isinstance(other, VProduct):
            return VProduct(*(self.objects+other.objects))
        return VProduct(*(self.objects+[other]))

    def __rmul__(self, other):
        if isinstance(other, VProduct):
            return VProduct(*(other.objects+self.objects))

        return VProduct(other, *self.objects)

class Transformation(Indexed):
    def __init__(self, name, *indices, **kwargs):
        super(Variable, self).__init__(name, *indices, **kwargs)

    def __mul__(self, other):
        return VProduct(self, other)

        # if isinstance(other, VSum):
            # return VProduct(*(self.objects+other.objects))


# class VSum(Indexed):
    # def __init__(self, *objects, **kwargs):
        # name = kwargs.pop('name', '')
        # self.objects=list(objects)
        # super(VSum, self).__init__(name, collect=[o.indices for o in self.objects])

    # def __str__(self):
        # return '( {} )'.format( ' + '.join(str(o) for o in self.objects) )

    # def __add__(self, other):
        # if isinstance(other, VSum):
            # return VSum(*(self.objects+other.objects))

        # return VSum(*(self.objects+[other]))

    # def __radd__(self, other):
        # if isinstance(other, VSum):
            # return VSum(*(other.objects+self.objects))

        # return VSum(other, *self.objects)

