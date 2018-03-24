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

    def singular(self):
        return not bool(self.indices)

    def plural(self):
        return bool(self.indices)

    def __eq__(self, other):
        if not isinstance(other, Indices):
            return False
        return self.indices==other.indices

    def reduce(self, *indices):
        if not set(indices).issubset(self.indices):
            raise Exception( "Indices.reduce should be called on a subset of indices" )

        return Indices(*(set(self.indices)-set(indices)))

    def ident(self):
        return '_'.join(self.indices)

class Indexed(Indices):
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        super(Indexed,self).__init__(*indices, **kwargs)

    def __add__(self, other):
        raise Exception('not implemented')

    def __str__(self):
        if self.indices:
            return '{}[{}]'.format(self.name, Indices.__str__(self))
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

    def reduce(self, newname, *indices):
        return Indexed(newname, Indices.reduce(self, *indices))

    def walk(self, yieldself=False, level=0):
        yield level, self

class Variable(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, Transformation):
            return WeightedTransformation('?', self, other)

        return VProduct('?', self, other)

    def __call__(self, *targs):
        return Transformation(self.name, self, targs=targs)

class VProduct(Variable):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for VarProduct')

        self.objects = []
        for o in objects:
            if not isinstance(o, Variable):
                raise Exception('Expect Variable instance')

            if isinstance(o, VProduct):
                self.objects+=o.objects
            else:
                self.objects.append(o)

        super(VProduct, self).__init__(name, *objects, **kwargs)

    def estr(self, expand=100):
        if expand:
            expand = expand-1
            return '{}'.format( ' * '.join(o.estr(expand) for o in self.objects) )
        else:
            return self.__str__()

    def walk(self, yieldself=False, level=0):
        if yieldself:
            yield level, self
        level+=1
        for o in self.objects:
            for sub in  o.walk(yieldself, level):
                yield sub

class Transformation(Indexed):
    def __init__(self, name, *args, **kwargs):
        targs = ()
        if '|' in args:
            idx = args.index('|')
            args, targs = args[:idx], args[idx+1:]

        arguments = list(targs) + list(kwargs.pop('targs', ()))
        self.arguments=[]
        for arg in arguments:
            if isinstance(arg, str):
                arg = Transformation(arg)
            elif not isinstance(arg, Transformation):
                raise Exception('Transformation argument should be another Transformation')
            self.arguments.append(arg)

        super(Transformation, self).__init__(name, *(list(args)+self.arguments), **kwargs)

    def __str__(self):
        return '{}({})'.format(Indexed.__str__(self), ', '.join(str(a) for a in self.arguments))

    def __mul__(self, other):
        if isinstance(other, (Variable, WeightedTransformation)):
            return WeightedTransformation('?', self, other)

        return TProduct('?', self, other)

    def __add__(self, other):
        return TSum('?', self, other)

class TProduct(Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TProduct')

        self.objects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance')

            if isinstance(o, TProduct):
                self.objects+=o.objects
            else:
                self.objects.append(o)

        super(TProduct, self).__init__(name, *objects, **kwargs)

    def estr(self, expand=100):
        if expand:
            expand = expand-1
            return ' * '.join(o.estr(expand) for o in self.objects)
        else:
            return self.__str__()

    def walk(self, yieldself=False, level=0):
        if yieldself:
            yield level, self
        level+=1
        for o in self.objects:
            for sub in o.walk(yieldself, level):
                yield sub

class TSum(Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TSum')

        self.objects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance')

            if isinstance(o, TSum):
                self.objects+=o.objects
            else:
                self.objects.append(o)

        super(TSum, self).__init__(name, *objects, **kwargs)

    def estr(self, expand=100):
        if expand:
            expand = expand-1
            return '({})'.format(' + '.join(o.estr(expand) for o in self.objects))
        else:
            return self.__str__()

    def walk(self, yieldself=False, level=0):
        if yieldself:
            yield level, self
        level+=1
        for o in self.objects:
            for sub in o.walk(yieldself, level):
                yield sub

class WeightedTransformation(Transformation):
    object, weight = None, None
    def __init__(self, name, *objects, **kwargs):
        for other in objects:
            if isinstance(other, WeightedTransformation):
                self.object = self.object*other.object if self.object is not None else other.object
                self.weight = self.weight*other.weight if self.weight is not None else other.weight
            elif isinstance(other, Variable):
                self.weight = self.weight*other if self.weight is not None else other
            elif isinstance(other, Transformation):
                self.object = self.object*other if self.object is not None else other
            else:
                raise Exception( 'Unsupported type' )

        super(WeightedTransformation, self).__init__(name, self.object, self.weight, targs=(), **kwargs)

    def estr(self, expand=100):
        if expand:
            expand = expand-1
            return '{:s} * {:s}'.format( self.weight.estr(expand), self.object.estr(expand) )
        else:
            return self.__str__()

    def __mul__(self, other):
        return WeightedTransformation('?', self, other)

    def walk(self, yieldself=False, level=0):
        if yieldself:
            yield level, self
        level+=1
        for sub in self.weight.walk(yieldself, level):
            yield sub
        for sub in self.object.walk(yieldself, level):
            yield sub

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

