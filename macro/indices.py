#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict

class Indices(object):
    def __init__(self, *indices, **kwargs):
        collect = kwargs.pop('collect', [])
        indices = set(indices)
        for sub in collect:
            indices|=set(sub.indices)
        self.indices=list(sorted(indices))

    def __str__(self):
        return ', '.join( self.indices )

    def __add__(self, other):
        return Indices(*(self.indices+other.indices))

    def __bool__(self):
        return bool(self.indices)

    __nonzero__=__bool__

    def __eq__(self, other):
        return self.indices==other.indices

class Indexed(object):
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        self.indices = Indices(*indices, **kwargs)

    def __str__(self):
        if self.indices:
            return '{}[{}]'.format(self.name, str(self.indices))
        else: return self.name

    def seq(self):
        return [self]

    def __eq__(self, other):
        if self.name!=other.name:
            return False
        return self.indices==other.indices

    def has_weight(self, weight):
        return False

    def has_weight(self, weight):
        return self==weight

    def __mul__(self, other):
        return Product(self, other)

    def __add__(self, other):
        return Sum(self, other)

class Sum(Indexed):
    def __init__(self, *objects, **kwargs):
        name = kwargs.pop('name', '')
        self.objects=list(objects)
        super(Sum, self).__init__(name, collect=[o.indices for o in self.objects])

    def __str__(self):
        return '( {} )'.format( ' + '.join(str(o) for o in self.objects) )

    def __add__(self, other):
        if isinstance(other, Sum):
            return Sum(*(self.objects+other.objects))

        return Sum(*(self.objects+[other]))

    def __radd__(self, other):
        if isinstance(other, Sum):
            return Sum(*(other.objects+self.objects))

        return Sum(other, *self.objects)

class Product(Indexed):
    def __init__(self, *objects, **kwargs):
        name = kwargs.pop('name', '')
        self.objects=list(objects)
        super(Product, self).__init__(name, collect=[o.indices for o in self.objects])

    def __str__(self):
        return '{}'.format( ' * '.join(str(o) for o in self.objects) )

    def __mul__(self, other):
        if isinstance(other, Sum):
            return Product(*(self.objects+other.objects))

        return Product(*(self.objects+[other]))

    def __rmul__(self, other):
        if isinstance(other, Product):
            return Product(*(other.objects+elf.objects))

        return Product(other, *self.objects)

    # def has_weight(self, weight):
        # for w in self.weights:
            # if w.has_weight(weight):
                # return True
        # return False

    # def remove_weight(self, weight):
        # for i, w in enumerate(self.weights):
            # if w==weight:
                # del self.weights[i]
                # break

jac    = Indexed( 'Jac' )
dnorm  = Indexed( 'detnorm', 'd' )
prod=jac*dnorm
s=jac+dnorm
print(jac, dnorm)
print(prod)
print(s)
print(s+prod)
print(s*prod)
print(prod+prod)
print(prod*prod)
print(s+s)
print(s*s)
print(jac+prod)
print(prod*jac)
print(jac+s)
print(jac*s)

print( ( prod+prod )*s )

# class Sum(Indexed):
    # def __init__(self, name, *objects):
        # self.objects = list(objects)
        # super(Sum, self).__init__(name, collect=[o.indices for o in self.objects])

    # def __str__(self):
        # return '( {} )[{}]'.format( ' + '.join( (str(o) for o in self.objects ) ), str(self.indices) )

    # def common_weight(self, weight):
        # for obj in self.objects:
            # if not obj.has_weight(weight):
                # return False
        # return True

    # def extract(self):
        # common = []
        # for weight in self.objects[0].weights:
            # if self.common_weight( weight ):
                # common.append(weight)
                # for obj in self.objects:
                    # obj.remove_weight( weight )
        # return common

    # def open(self):
        # newobjects = []
        # for obj in self.objects:
            # if hasattr(obj, 'open'):
                # opened = obj.open().objects
                # newobjects+=opened
            # else:
                # newobjects+=[obj]

        # self.objects=newobjects
        # return self

# class Product(Indexed):
    # def __init__(self, name, *objects):
        # self.objects = []
        # for obj in objects:
            # self.objects+=obj.seq()
        # super(Product, self).__init__(name, collect=[o.indices for o in self.objects])

    # def __str__(self):
        # return '{}: {}'.format(Indexed.__str__(self), ' '.join(str(o) for o in self.objects))

    # def extract(self):
        # for i in range(1, len(self.objects)+1):
            # obj = self.objects[-i]
            # if isinstance(obj, Sum):
                # common = obj.extract()
                # if common:
                    # self.objects=self.objects[:-i]+common+self.objects[-i:]

    # def open(self):
        # for i, obj in enumerate(self.objects):
            # if isinstance(obj, Sum):
                # before, after = self.objects[:i], self.objects[i+1:]

                # sums = []
                # for j, obj in enumerate(obj.objects):
                    # psum = Product( self.name+'_'+str(j), *(before+[obj]+after) )
                    # sums.append(psum)
                # newpsum = Sum(self.name, *sums)

                # return newpsum.open()
        # return self


# # spec = Indexed('spectrum', 'i')
# # offeq = Indexed('offeq', 'i', 'r')
# # print(spec)
# # print(offeq)
# # print(offeq.indices+offeq.indices)

# indices = OrderedDict()
# indices['c'] = [ 'comp'+str(i) for i in range (1, 4) ]
# indices['i'] = [ 'U5', 'U8', 'Pu9', 'Pu1' ]
# indices['r'] = [ 'DB'+str(i) for i in range(1,3)]+['LA'+str(i) for i in range(1,5) ]
# indices['d'] = [ 'AD'+str(i) for i in [ 11, 12, 21, 22, 31, 32, 33, 34 ] ]

# print( 'Indices:' )
# for k, v in indices.items():
    # print('   ', k, v)
# print()

# speci = IWeighted( 'Isotoper', [['ffrac', 'i', 'r'], ['power', 'r']], ('Isotope', 'i') )
# offeq = IWeighted( 'Offeqr',   [['ffrac', 'i', 'r'], ['offeq_norm', 'i', 'r'], ['power', 'r']], ('Offeq', 'i') )
# snf   = IWeighted( 'Snf',      [['snf_norm', 'r'], ['power', 'r']], ('Snf', 'i') )
# comp0 = IWeighted( 'Comp0',    [['oscprobw0',]], ('OscProbItem0',) )
# compi = IWeighted( 'CompI',    [['oscprobw', 'c']], ('OscProbItem', 'c') )
# integr = Indexed( 'Integrate' )
# xsec   = Indexed( 'Xsec' )
# jac    = Indexed( 'Jac' )
# dnorm  = Indexed( 'detnorm', 'd' )
# baseline  = Indexed( 'baseline', 'd', 'r' )
# rnorm  = Indexed( 'reacnorm', 'r' )
# print( speci )
# print( offeq )
# print( snf )
# print( comp0 )
# print( compi )
# print()

# spec=Sum('spec', speci, offeq, snf)
# comp=Sum('comp', comp0, compi)
# ps = Product( 'prod', dnorm, rnorm, baseline, integr, xsec, jac, comp, spec )

# print( 'Raw' )
# print( ps )

# print( 'Extract' )
# ps.extract()
# print(ps)

# print( 'Arrange' )
# op=ps.open()
# print(op)
