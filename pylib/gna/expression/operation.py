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
    order_from=None
    def __init__(self, operation, *indices, **kwargs):
        self.operation=operation
        self.nindex_to_reduce = NIndex(*indices, order=self.order)
        TCall.__init__(self, undefinedname)
        NestedTransformation.__init__(self)

    def __str__(self):
        return '{}{{{:s}}}'.format(Indexed.__str__(self), self.nindex_to_reduce.comma_list())

    def __call__(self, *args):
        if self.call_lock:
            raise Exception('May call Operation only once')
        self.call_lock=True

        self.set_objects(*args)
        self.set_indices(*args, ignore=self.nindex_to_reduce.indices.keys())
        return self

    def guessname(self, lib={}, save=False):
        for o in self.objects:
            o.guessname(lib, save)

        if self.name is not undefinedname:
            if not self.label:
                for cfg in lib.values():
                    if cfg['name']!=self.name:
                        continue
                    newlabel = cfg.get('label', None)
                    if newlabel:
                        self.label=newlabel
            return self.name

        cname = TCall.guessname(self, lib=lib)
        newname='{}:{}|{}'.format(self.operation, self.nindex_to_reduce.ident(), cname)

        label=None

        if newname in lib:
            libentry = lib[newname]
            newname = libentry['name']
            label   = libentry.get('label', None)
        else:
            newname = '{}{}'.format(self.text_operator, cname)

        if save:
            self.name = newname
            self.set_label(label)

        return newname

    @methodname
    def require(self, context):
        IndexedContainer.require(self, context)

    @call_once
    def bind(self, context):
        printl_debug('bind (operation) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

        if not self.tinit:
            return

        with nextlevel():
            printl_debug('def (operation)', str(self))

            nitems = self.nindex_to_reduce.get_size() * len(self.objects)
            with nextlevel():
                for freeidx in self.nindex.iterate():
                    if nitems>1:
                        # In case there are multiple elements: produce sum/product/etc
                        tobj, newout = self.new_tobject(freeidx)
                        context.set_output(newout, self.name, freeidx)
                        with nextlevel():
                            for opidx in self.nindex_to_reduce.iterate():
                                fullidx = freeidx+opidx
                                for i, obj in enumerate(self.objects):
                                    output = obj.get_output(fullidx, context)
                                    inp  = tobj.add_input('%02d'%i)
                                    context.set_input(inp, self.name, fullidx, clone=i)
                                    output >> inp
                    elif nitems==1:
                        # In case there is only one element: pass it as is
                        for opidx in self.nindex_to_reduce.iterate():
                            fullidx = freeidx+opidx
                            output = self.objects[0].get_output(fullidx, context)
                            context.set_output(output, self.name, freeidx)
                            break
                    else:
                        raise Exception('Index is not iterable')

class OSum(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'sum', *indices, **kwargs)
        self.set_operator( ' Σ ', text='sum_' )

        import ROOT as R
        self.set_tinit( R.Sum )

    @call_once
    def bind(self, context):
        # Process sum of weigtedsums
        if self.expandable and len(self.objects)==1 and isinstance(self.objects[0], WeightedTransformation) and self.objects[0].expandable:
            return self.bind_wsum(context)

        Operation.bind(self, context)

    def bind_wsum(self, context):
        import ROOT as R
        self.set_tinit( R.WeightedSum )

        printl_debug('bind (osum: weighted) {}:'.format(type(self).__name__), str(self) )

        weight    = self.objects[0].weight
        subobject = self.objects[0].object

        self.objects_orig = self.objects
        self.objects = [ subobject ]
        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

        from gna.constructors import stdvector
        with nextlevel():
            for freeidx in self.nindex.iterate():
                rindices = [ridx for ridx in self.nindex_to_reduce.iterate()]
                names    = stdvector([(ridx+freeidx).current_format('{autoindex}') for ridx in rindices])
                weights  = stdvector([weight.current_format(ridx+freeidx) for ridx in rindices])

                tobj, newout = self.new_tobject(freeidx, weights, names, weight_label=weight.name)
                context.set_output(newout, self.name, freeidx)

                for i, (name, reduceidx) in enumerate(zip(names, rindices)):
                    fullidx = freeidx+reduceidx
                    inp = context.set_input(tobj.sum.inputs[name], self.name, freeidx, clone=i)
                    output = subobject.get_output(fullidx, context)
                    output >> inp

            # from gna.constructors import stdvector
            # labels  = stdvector([self.object.name])
            # printl_debug('connect (weighted)')
            # for idx in self.nindex.iterate():
                # wname = self.weight.current_format(idx)
                # weights = stdvector([wname])

                # with context.ns:
                    # tobj, newout = self.new_tobject( self.current_format(idx), labels, weights )
                # inp = tobj.transformations[0].inputs[0]
                # context.set_output(newout, self.name, idx)
                # context.set_input(inp, self.name, idx)
                # out = self.object.get_output(idx, context)
                # out >> inp

placeholder=['placeholder']
class OProd(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'prod', *indices, **kwargs)
        self.set_operator( ' Π ', text='prod_' )

        import ROOT as R
        self.set_tinit( R.GNA.GNAObjectTemplates.ProductT('double') )

class OConcat(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'concat', *indices, **kwargs)
        self.set_operator( ' ⊕ ', text='concat_' )

        import ROOT as R
        self.set_tinit( R.Concat )

class OInverse(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'inv', *indices, **kwargs)
        self.set_operator( ' / ', text='inv_' )

        import ROOT as R
        self.set_tinit( R.Inverse )


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
        self.set_operator( ' ∫ ', text='accumulate_'  )

    @call_once
    def bind(self, context):
        if self.bound:
            return

        import ROOT as R
        IndexedContainer.bind(self, context, connect=False)
        obj, = self.objects
        ns = context.namespace()
        from gna.env import ExpressionsEntry
        from gna import constructors as C
        for it in self.nindex.iterate():
            out = obj.get_output(it, context)
            varname = self.current_format(it)

            head, tail = varname.rsplit('.', 1)
            cns = ns(head)

            label = 'sum of {}'.format(obj.current_format(it))
            arrsum = C.ArraySum(tail, out, ns=cns, labels=label)
            var = cns[tail].get()
            var.setLabel(label)
            self.arrsums.append(arrsum)

        self.bound = True

class AccumulateTransformation(IndexedContainer, Transformation):
    bound = False
    def __init__(self, *args, **kwargs):
        self.arrsums = []
        if len(args)>1:
            raise Exception('accumulate() supports only 1 argument')
        if not isinstance(args[0], Transformation):
            raise Exception('the only argument of accumulate() should be an object, not variable')

        IndexedContainer.__init__(self, *args)
        Transformation.__init__(self, undefinedname, *self.objects)
        self.set_operator( ' ∫ ', text='Accumulate_', prefix='Accumulate_'  )

    @call_once
    def bind(self, context):
        if self.bound:
            return

        import ROOT as R
        IndexedContainer.bind(self, context, connect=False)
        obj, = self.objects
        ns = context.namespace()
        from gna.env import ExpressionsEntry
        from gna import constructors as C
        for it in self.nindex.iterate():
            out = obj.get_output(it, context)
            varname = self.current_format(it)

            head, tail = varname.rsplit('.', 1)
            cns = ns(head)

            label = 'sum of {}'.format(obj.current_format(it))
            arrsum = C.ArraySum(out, ns=cns, labels=label)
            output = arrsum.arrsum.sum
            context.set_output(output, self.name, it)

            self.arrsums.append(arrsum)

        self.bound = True

class OSelect1(TCall):
    """Substitute a single index with another"""
    __metaclass__ = OperationMeta
    call_lock=False
    order_from=None
    def __init__(self, index, value, **kwargs):
        self.operation='select1'
        index.set_current(value)
        self._index_to_reduce = index
        self._nindex_to_reduce=None
        self._index_value = value
        TCall.__init__(self, undefinedname)

    def __str__(self):
        return '{}{{{:s}}}'.format(Indexed.__str__(self), self._index_to_reduce.short)

    def __call__(self, *args):
        if self.call_lock:
            raise Exception('May call Operation only once')
        self.call_lock=True

        self.set_objects(*args)
        self.set_indices(*args, ignore=self._index_to_reduce.short)
        self._nindex_to_reduce = NIndex(self._index_to_reduce, order=self.nindex.order)
        return self

    def guessname(self, lib={}, save=False):
        for o in self.objects:
            o.guessname(lib, save)

        if self.name is not undefinedname:
            if not self.label:
                for cfg in lib.values():
                    if cfg['name']!=self.name:
                        continue
                    newlabel = cfg.get('label', None)
                    if newlabel:
                        self.label=newlabel
            return self.name

        cname = TCall.guessname(self, lib=lib)
        newname='{}:{}|{}'.format(self.operation, self._indices_to_replace[1].ident(), cname)

        label=None

        if newname in lib:
            libentry = lib[newname]
            newname = libentry['name']
            label   = libentry.get('label', None)
        else:
            newname = '{}{}'.format(self.text_operator, cname)

        if save:
            self.name = newname
            self.set_label(label)

        return newname

    @methodname
    def require(self, context):
        IndexedContainer.require(self, context)

    @call_once
    def bind(self, context):
        printl_debug('bind (operation) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

        with nextlevel():
            printl_debug('def (operation)', str(self))
            with nextlevel():
                for freeidx in self.nindex.iterate():
                    idxin = freeidx + self._nindex_to_reduce
                    output = self.objects[0].get_output(idxin, context)
                    context.set_output(output, self.name, freeidx)

def bracket(obj):
    obj.expandable = False
    return obj

def expand(obj):
    obj.expandable = True
    return obj
