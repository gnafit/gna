#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.compound import *

class OperationMeta(type):
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = args,
        return cls(*args)

class Operation(TCall,NestedTransformation):
    __metaclass__ = OperationMeta
    call_lock=False
    def __init__(self, operation, *indices, **kwargs):
        self.operation=operation
        self.indices_to_reduce = NIndex(*indices)
        TCall.__init__(self, undefinedname)
        NestedTransformation.__init__(self)

    def __str__(self):
        return '{}{{{:s}}}'.format(Indexed.__str__(self), self.indices_to_reduce)

    def __call__(self, *args):
        if self.call_lock:
            raise Exception('May call Operation only once')
        self.call_lock=True

        self.set_objects(*args)
        self.set_indices(*args, ignore=self.indices_to_reduce.indices.keys())
        return self

    def guessname(self, lib={}, save=False):
        for o in self.objects:
            o.guessname(lib, save)

        cname = TCall.guessname(self)
        newname='{}:{}|{}'.format(self.operation, self.indices_to_reduce.ident(), cname)

        if newname in lib:
            libentry = lib[newname]
            newname = libentry['name']
            label   = libentry.get('label', None)

            if save:
                self.name = newname
                self.set_label(label)

        return newname

    @methodname
    def require(self, context):
        IndexedContainer.require(self, context)

    @call_once
    def bind(self, context):
        printl('bind (operation) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

        if not self.tinit:
            return

        with nextlevel():
            printl('def (operation)', str(self))
            with nextlevel():
                for freeidx in self.indices.iterate():
                    tobj, newout = self.new_tobject(self.current_format(freeidx))
                    context.set_output(newout, self.name, freeidx)
                    with nextlevel():
                        for opidx in self.indices_to_reduce.iterate():
                            fullidx = freeidx+opidx
                            for i, obj in enumerate(self.objects):
                                output = obj.get_output(fullidx, context)
                                inp  = tobj.add_input('%02d'%i)
                                context.set_input(inp, self.name, fullidx, clone=i)
                                inp(output)

class OSum(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'sum', *indices, **kwargs)
        self.set_operator( ' Σ ' )

        import ROOT as R
        self.set_tinit( R.Sum )

    @call_once
    def bind(self, context):
        # Process sum of weigtedsums
        if len(self.objects)==1 and isinstance(self.objects[0], WeightedTransformation):
            return self.bind_wsum(context)

        Operation.bind(self, context)

    def bind_wsum(self, context):
        import ROOT as R
        self.set_tinit( R.WeightedSum )

        printl('bind (osum: weighted) {}:'.format(type(self).__name__), str(self) )

        weight    = self.objects[0].weight
        subobject = self.objects[0].object

        self.objects_orig = self.objects
        self.objects = [ subobject ]
        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

        from constructors import stdvector
        with nextlevel():
            for freeidx in self.indices.iterate():
                rindices = [ridx for ridx in self.indices_to_reduce.iterate()]
                names    = stdvector([(ridx+freeidx).current_format('{autoindexnd}') for ridx in rindices])
                weights  = stdvector([weight.current_format(ridx+freeidx) for ridx in rindices])

                tobj, newout = self.new_tobject(freeidx, names, weights, weight_label=weight.name)
                context.set_output(newout, self.name, freeidx)

                for i, (name, reduceidx) in enumerate(zip(names, rindices)):
                    fullidx = freeidx+reduceidx
                    inp = context.set_input(tobj.sum.inputs[name], self.name, freeidx, clone=i)
                    output = subobject.get_output(fullidx, context)
                    inp(output)

            # from constructors import stdvector
            # labels  = stdvector([self.object.name])
            # printl('connect (weighted)')
            # for idx in self.indices.iterate():
                # wname = self.weight.current_format(idx)
                # weights = stdvector([wname])

                # with context.ns:
                    # tobj, newout = self.new_tobject( self.current_format(idx), labels, weights )
                # inp = tobj.transformations[0].inputs[0]
                # context.set_output(newout, self.name, idx)
                # context.set_input(inp, self.name, idx)
                # out = self.object.get_output(idx, context)
                # inp(out)

placeholder=['placeholder']
class OProd(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'prod', *indices, **kwargs)
        self.set_operator( ' Π ' )

        import ROOT as R
        self.set_tinit( R.Product )

class Accumulate(IndexedContainer, Variable):
    bound = False
    def __init__(self, name, *args, **kwargs):
        self.arrsums = []
        if len(args)>1:
            raise Exception('accumulate() supports only 1 argument')
        if not isinstance(args[0], Transformation):
            raise Exception('the only argument of accumulate() should be an object, not variable')

        IndexedContainer.__init__(self, *args)
        Variable.__init__(self, name, *self.objects)
        self.set_operator( ' ∫ ' )

    @call_once
    def bind(self, context):
        if self.bound:
            return

        import ROOT as R
        IndexedContainer.bind(self, context, connect=False)
        obj, = self.objects
        ns = context.namespace()
        from gna.env import ExpressionsEntry
        for it in self.indices.iterate():
            out = obj.get_output(it, context)
            varname = self.current_format(it)

            head, tail = varname.rsplit('.', 1)
            cns = ns(head)
            arrsum = R.ArraySum(tail, out, ns=cns)
            var = cns[tail].get()
            var.setLabel('sum of {}'.format(obj.current_format(it)))
            self.arrsums.append(arrsum)

        self.bound = True

