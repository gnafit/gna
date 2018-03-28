# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict

class Index(object):
    def __init__(self, short, name, variants):
        self.short = short
        self.name  = name
        self.variants = variants

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
            elif isinstance(idx, Index):
                self.indices.add(idx.short)
            else:
                raise Exception( 'Unsupported index type '+type(idx).__name__ )

        ignore = kwargs.pop('ignore', None)
        if ignore:
            self.indices.discard(*ignore.indices)

        self.indices=list(sorted(self.indices))

        if kwargs:
            raise Exception('Unparsed kwargs: {:s}'.format(kwargs))

    def __str__(self):
        return ', '.join( self.indices )

    def __add__(self, other):
        return Indices(self, other)

    def __bool__(self):
        return bool(self.indices)

    __nonzero__ = __bool__

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
    indices_locked=False
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        self.set_indices(*indices, **kwargs)

    def set_indices(self, *indices, **kwargs):
        self.indices=Indices(*indices, **kwargs)
        if indices:
            self.indices_locked=True

    def __getitem__(self, args):
        if self.indices_locked:
            raise Exception('May not modify already declared indices')
        if not isinstance(args, tuple):
            args = args,
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

    def reduce(self, newname, *indices):
        return Indexed(newname, self.indices.reduce(*indices))

    def walk(self, yieldself=False, level=0, operation=''):
        yield self, level, operation

    def ident(self, **kwargs):
        if self.name=='?':
            return self.guessname(**kwargs)
        return self.name

    def ident_full(self, **kwargs):
        return '{}:{}'.format(self.ident(**kwargs), self.indices.ident(**kwargs))

    def guessname(self, *args, **kwargs):
        return '?'

    def dump(self, yieldself=False):
        for i, (obj, level, operator) in enumerate(self.walk(yieldself)):
            print( i, level, '    '*level, operator, obj )

class IndexedContainer(object):
    objects = None
    operator='.'
    left, right = '', ''
    def __init__(self, *objects):
        self.set_objects(*objects)

    def set_objects(self, *objects):
        self.objects = list(objects)

    def walk(self, yieldself=False, level=0, operation=''):
        if yieldself:
            yield self, level, operation
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
        for o in self.objects:
            o.guessname(lib, save)

        newname = '{expr}'.format(
                    expr = self.operator.strip().join(sorted(o.ident(lib=lib, save=save) for o in self.objects)),
                    )

        newnameu = '{expr}'.format(
                    expr = self.operator.strip().join(o.ident(lib=lib, save=save) for o in self.objects),
                     )

        variants=[newnameu, newname]
        for nn in tuple(variants):
            variants.append(nn+':'+self.indices.ident())

        guessed = False
        for var in variants:
            if var in lib:
                guessed = lib[var]['name']
                break

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

    def nonempty(self):
        return bool(self.objects)

class Variable(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, Transformation):
            return WeightedTransformation('?', self, other)

        return VProduct('?', self, other)

    def __call__(self, *targs):
        return TCall(self.name, self, targs=targs)

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
        super(Transformation, self).__init__(name, *args, **kwargs)

    def __str__(self):
        return '{}()'.format(Indexed.__str__(self))

    def __mul__(self, other):
        if isinstance(other, (Variable, WeightedTransformation)):
            return WeightedTransformation('?', self, other)

        return TProduct('?', self, other)

    def __add__(self, other):
        return TSum('?', self, other)

class TCall(IndexedContainer, Transformation):
    def __init__(self, name, *args, **kwargs):
        targs = ()
        if '|' in args:
            idx = args.index('|')
            args, targs = args[:idx], args[idx+1:]

        targs = list(targs) + list(kwargs.pop('targs', ()))

        objects = []
        for arg in targs:
            if isinstance(arg, str):
                arg = Transformation(arg)
            elif not isinstance(arg, Transformation):
                raise Exception('Arguments argument should be another Transformation')
            objects.append(arg)

        IndexedContainer.__init__(self, *objects)
        Transformation.__init__(self,name, *(list(args)+list(objects)), **kwargs)
        self.set_operator( ', ', '(', ')' )

    def __str__(self):
        return '{}({:s})'.format(Indexed.__str__(self), '...' if self.objects else '' )

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
        Transformation.__init__(self, name, self.weight, self.object, **kwargs)

        self.set_operator( ' * ' )

    def __mul__(self, other):
        return WeightedTransformation('?', self, other)

class OperationMeta(type):
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = args,
        return cls(*args)

class Operation(TCall):
    __metaclass__ = OperationMeta
    call_lock=False
    def __init__(self, name, *indices, **kwargs):
        self.reduced_indices = Indices(*indices)
        TCall.__init__(self, name)

    def __str__(self):
        return '{}{{{:s}}}({:s})'.format(Indexed.__str__(self), self.reduced_indices, '...' if self.objects else '' )

    def __call__(self, *args):
        if self.call_lock:
            raise Exception('May call Operation only once')
        self.call_lock=True

        self.set_objects(*args)
        self.set_indices(*args, ignore=self.reduced_indices)
        return self

class OSum(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'sum', *indices, **kwargs)

class OProd(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'prod', *indices, **kwargs)

class VTContainer(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(VTContainer, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        newvar = Variable(key)
        self[key] = newvar
        return newvar

class Expression(object):
    operations = dict(sum=OSum, prod=OProd)
    def __init__(self, expression, indices):
        self.expression_raw = expression
        self.expression = self.preprocess( self.expression_raw )

        self.globals=VTContainer(self.operations)
        self.indices=OrderedDict()
        self.defindices(indices)


    def preprocess(self, expression):
        n = expression.count('|')
        if n:
            expression = expression.replace('|', '(')
            expression+=')'*n
        return expression

    def parse(self):
        self.tree=eval(self.expression, self.globals)

    def __str__(self):
        return self.expression_raw

    def __repr__(self):
        return 'Expression("{}")'.format(self.expression_raw)

    def newindex(self, short, name, *variants):
        idx = self.indices[short] = Index(short, name, variants)
        self.globals[short] = idx

    def defindices(self, defs):
        for d in defs:
            self.newindex(*d)

