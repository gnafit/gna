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
    def __init__(self, name, *indices, **kwargs):
        self.indices_to_reduce = NIndex(*indices)
        TCall.__init__(self, name)
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

        newname=self.name+':'+self.indices_to_reduce.ident()
        if newname in lib:
            newname = lib[newname]['name']

            if save:
                self.name = newname
        return newname

    def make_object(self, *args, **kwargs):
        raise Exception('Unimplemented method called')

    def build(self, context):
        printl('build (operation) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            IndexedContainer.build(self, context, connect=False)

        if not self.tinit:
            return

        with nextlevel():
            printl('connect (operation)', str(self))
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
        self.set_operator( ' ++ ' )

        import ROOT as R
        self.set_tinit( R.Sum )

    def build(self, context):
        # Process sum of weigtedsums
        if len(self.objects)==1 and isinstance(self.objects[0], WeightedTransformation):
            return self.build_wsum(context)

        Operation.build(self, context)

    def build_wsum(self, context):
        import ROOT as R
        self.set_tinit( R.WeightedSum )

        printl('build (osum: weighted) {}:'.format(type(self).__name__), str(self) )

        weight    = self.objects[0].weight
        subobject = self.objects[0].object

        self.objects_orig = self.objects
        self.objects = [ subobject ]
        with nextlevel():
            IndexedContainer.build(self, context, connect=False)

        from constructors import stdvector
        with nextlevel():
            for freeidx in self.indices.iterate():
                rindices = [ridx for ridx in self.indices_to_reduce.iterate()]
                names    = stdvector([(ridx+freeidx).current_format('{autoindexnd}') for ridx in rindices])
                weights  = stdvector([weight.current_format(ridx+freeidx) for ridx in rindices])

                tobj, newout = self.new_tobject(self.current_format(freeidx), names, weights)
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
        self.set_operator( ' ** ' )

        import ROOT as R
        self.set_tinit( R.Product )
