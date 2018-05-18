#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.simple import *

class IndexedContainer(object):
    objects = None
    operator='.'
    left, right = '', ''
    def __init__(self, *objects):
        self.set_objects(*objects)

    def set_objects(self, *objects):
        self.objects = list(objects)

    def walk(self, yieldself=False, operation=''):
        if yieldself or not self.objects:
            yield self, operation
        with nextlevel():
            for o in self.objects:
                for sub in  o.walk(yieldself, self.operator.strip()):
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

    def build(self, context, connect=True):
        if not self.objects:
            return

        printl('build (container) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            for obj in self.objects:
                context.check_outputs(obj)

            if not connect:
                return

            printl('connect (container)', str(self))
            with nextlevel():
                for idx in self.indices.iterate():
                    printl( 'index', idx )
                    with nextlevel():
                        nobj = len(self.objects)
                        for i, obj in enumerate(self.objects):
                            # if nobj==1:
                                # i=None
                            output = obj.get_output(idx, context)
                            input  = self.get_input(idx, context, clone=i)
                            input(output)

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

class NestedTransformation(object):
    tinit = None

    def __init__(self):
        self.tobjects = []

    def set_tinit(self, obj):
        self.tinit = obj

    def new_tobject(self, label, *args):
        newobj = self.tinit(*args)
        newobj.transformations[0].setLabel(label)
        self.tobjects.append(newobj)
        import ROOT as R
        return newobj, R.OutputDescriptor(newobj.single())

    def build(self, context):
        printl('build (nested) {}:'.format(type(self).__name__), str(self) )

        if self.tinit:
            with nextlevel():
                for idx in self.indices.iterate():
                    tobj, newout = self.new_tobject(idx.current_format('{name}{autoindex}', name=self.name))
                    context.set_output(newout, self.name, idx)
                    nobj = len(self.objects)
                    for i, obj in enumerate(self.objects):
                        # if nobj==1:
                            # i=None
                        inp = tobj.add_input('%02d'%i)
                        context.set_input(inp, self.name, idx, clone=i)

        with nextlevel():
            IndexedContainer.build(self, context)

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

    def estr(self, expand=100):
        if expand:
            expand-=1
            return '{fcn}{args}'.format(fcn=Indexed.__str__(self), args=IndexedContainer.estr(self, expand))
        else:
            return self.__str__()

    def build(self, context):
        printl('build (call) {}:'.format(type(self).__name__), str(self) )
        with nextlevel():
            Transformation.build(self, context)
            IndexedContainer.build(self, context)

class TProduct(NestedTransformation, IndexedContainer, Transformation):
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

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' * ' )
        import ROOT as R
        self.set_tinit( R.Product )

class TSum(NestedTransformation, IndexedContainer, Transformation):
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

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' + ', '(', ')' )
        import ROOT as R
        self.set_tinit( R.Sum )

class WeightedTransformation(NestedTransformation, IndexedContainer, Transformation):
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

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, self.weight, self.object)
        Transformation.__init__(self, name, self.weight, self.object, **kwargs)

        self.set_operator( ' * ' )

        import ROOT as R
        self.set_tinit( R.WeightedSum )

    def __mul__(self, other):
        return WeightedTransformation('?', self, other)

    def build(self, context):
        printl('build (weighted) {}:'.format(type(self).__name__), str(self) )
        with nextlevel():
            IndexedContainer.build(self, context, connect=False)

            from constructors import stdvector
            labels  = stdvector([self.object.name])
            printl('connect (weighted)')
            for idx in self.indices.iterate():
                wname = idx.current_format('{name}{autoindex}', name=self.weight.name)
                weights = stdvector([wname])
                tobj, newout = self.new_tobject( idx.current_format('{name}{autoindex}', name=self.name), labels, weights )
                inp = tobj.transformations[0].inputs[0]
                context.set_output(newout, self.name, idx)
                context.set_input(inp, self.name, idx)
                out = self.object.get_output(idx, context)
                inp(out)

