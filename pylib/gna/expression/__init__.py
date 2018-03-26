# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict

class Indices(object):
    def __init__(self, *indices, **kwargs):
        self.indices = set()

        for idx in indices:
            if isinstance(idx, Indexed):
                self.indices |= set(idx.indices.indices)
            elif isinstance(idx, Indices):
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
            raise Exception( "Indices.reduce should be called on a subset of indices, got {:s} in {:s}".format(indices, self.indices) )

        return Indices(*(set(self.indices)-set(indices)))

    def ident(self, **kwargs):
        return '_'.join(self.indices)

class Indexed(object):
    name=''
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        self.indices=Indices(*indices, **kwargs)

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

    def reduce(self, newname, *indices):
        return Indexed(newname, self.indices.reduce(*indices))

    def walk(self, yieldself=False, level=0, operation=''):
        yield level, self, operation

    def ident(self, **kwargs):
        if self.name=='?':
            return self.guessname(**kwargs)
        return self.name

    def ident_full(self, **kwargs):
        return '{}:{}'.format(self.ident(**kwargs), self.indices.ident(**kwargs))

    def guessname(self, **kwargs):
        return '?'

class IndexedContainer(object):
    objects = None
    operator='.'
    left, right = '', ''
    def __init__(self, *objects):
        self.objects = list(objects)

    def walk(self, yieldself=False, level=0, operation=''):
        if yieldself:
            yield level, self, operation+':'
        level+=1
        for o in self.objects:
            for sub in  o.walk(yieldself, level, self.operator.strip()):
                yield sub

    def set_operator(self, operator, left=None, right=None):
        self.operator=operator
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right

    def guessname(self, lib={}, save=False):
        newname = '{expr}'.format(
                    expr = self.operator.strip().join(sorted(o.ident(lib=lib, save=save) for o in self.objects)),
                    )

        newnamei = newname+':'+self.indices.ident()

        guessed = False
        if newname in lib:
            guessed = lib[newname]
        elif newnamei in lib:
            guessed = lib[newnamei]

        if guessed:
            newname = guessed
            if save:
                self.name = newname

        return newname

    def estr(self, expand=100):
        if expand:
            expand = expand-1
            return '{left}{expr}{right}'.format(
                    left = self.left,
                    expr = self.operator.join(o.estr(expand) for o in self.objects),
                    right= self.right
                    )
        else:
            return self.__str__()

class Variable(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, Transformation):
            return WeightedTransformation('?', self, other)

        return VProduct('?', self, other)

    def __call__(self, *targs):
        return Transformation(self.name, self, targs=targs)

class VProduct(IndexedContainer, Variable):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for VarProduct')

        newobjects = []
        for o in objects:
            if not isinstance(o, Variable):
                raise Exception('Expect Variable instance')

            if isinstance(o, VProduct):
                newobjects+=o.objects
            else:
                newobjects.append(o)

        IndexedContainer.__init__(self, *newobjects)
        Variable.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( '*' )

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

class TProduct(IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TProduct')

        newobjects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance')

            if isinstance(o, TProduct):
                newobjects+=o.objects
            else:
                newobjects.append(o)

        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' * ' )

class TSum(IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TSum')

        newobjects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance')

            if isinstance(o, TSum):
                newobjects+=o.objects
            else:
                newobjects.append(o)

        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' + ', '(', ')' )

class WeightedTransformation(IndexedContainer, Transformation):
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

        IndexedContainer.__init__(self, self.weight, self.object)
        Transformation.__init__(self, name, self.weight, self.object, targs=(), **kwargs)

        self.set_operator( ' * ' )

    def __mul__(self, other):
        return WeightedTransformation('?', self, other)

