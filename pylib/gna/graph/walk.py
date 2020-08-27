# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
from collections import deque, namedtuple, OrderedDict
import numpy as N
import types
from gna.bindings import provided_precisions

VariableEntry = namedtuple('VariableEntry', ['fullname', 'variable', 'depends_entry', 'taints_entry', 'depends_var', 'taints_var'])

class GraphWalker(object):
    def __init__(self, *args, **kwargs):
        self._include_only=kwargs.pop('include_only', None)

        self._entry_points = []
        for arg in args:
            if isinstance(arg, (list,tuple)):
                for subarg in arg:
                    self._add_entry_point(subarg)
            self._add_entry_point(arg)

        self._build_cache()

        ns = kwargs.pop('namespace', None)
        if ns:
            self.set_parameters(ns)

        assert not kwargs, 'kwargs contains unparsed arguments: {!s}'.format(kwargs)

    def _add_entry_point(self, arg):
        if isinstance(arg, (types.GeneratorType)):
            arg = list(arg)
        if not isinstance(arg, (list, tuple)):
            arg = [arg]

        for t in arg:
            for precision in provided_precisions:
                OutputHandle = R.TransformationTypes.OutputHandleT(precision)
                Handle = R.TransformationTypes.HandleT(precision, precision)
                SingleOutput = R.SingleOutputT(precision)
                if isinstance(t, OutputHandle):
                    entry = R.OpenOutputHandleT(precision,precision)(t).getEntry()
                    break
                elif isinstance(t, Handle):
                    entry = R.OpenHandleT(precision,precision)(t).getEntry()
                    break
                elif isinstance(t, SingleOutput):
                    entry = R.OpenOutputHandleT(precision,precision)(t.single()).getEntry()
                    break
            else:
                # raise TypeError('GNADot argument should be of type TransformationDescriptor/TransformationTypes::Handle/TransformationTypes::OutputHandle, got '+type(t).__name__)
                raise TypeError('Unsupported argument type '+type(t).__name__)

        self._entry_points.append(entry)

    def _propagate_forward(self, entry, queue, skip):
        for sink in entry.sinks:
            for source in sink.sources:
                self._add_to_queue(queue, source.entry)

            if skip:
                continue

            if sink.sources.size()==0:
                self.cache_sinks_open.append(sink)

    def _propagate_backward(self, entry, queue, skip):
        for source in entry.sources:
            sink = source.sink

            if sink:
                self._add_to_queue(queue, sink.entry)
            elif not skip:
                self.cache_sources_open.append(source)

    def _build_cache(self):
        self.skipped_entries=[]
        self.cache_entries=[]
        self.cache_sources=[]
        self.cache_sinks=[]
        self.cache_variables=OrderedDict()

        self.cache_sources_open=[]
        self.cache_sinks_open=[]

        queue=self._add_to_queue(deque(), *self._entry_points)
        while queue:
            entry = queue.popleft()

            skip = self._if_skip_entry(entry)
            if skip:
                self.skipped_entries.append(entry)
            else:
                self.cache_entries.append(entry)
                self.cache_sources.extend((inp for inp in entry.sources if not inp in self.cache_sources))
                self.cache_sinks.extend(  (inp for inp in entry.sinks   if not inp in self.cache_sinks))

            self._propagate_forward(entry, queue, skip)
            self._propagate_backward(entry, queue, skip)

    def _add_to_queue(self, queue, *entries):
        for entry in entries:
            if entry in queue or entry in self.cache_entries or entry in self.skipped_entries:
                continue

            queue.append(entry)

        return queue

    def _if_skip_entry(self, entry):
        if not self._include_only:
            return False

        label=entry.attrs['_label']

        for p in self._include_only:
            if p in label:
                return False

        return True


    def set_parameters(self, ns):
        from gna import env
        self.cache_variables=OrderedDict()
        for (name, par) in ns.walknames():
            if isinstance(par, (str, env.ExpressionsEntry)):
                continue
            var = par.getVariable()
            # print('walk', name, var.hash())
            self.cache_variables[var.hash()] = VariableEntry(name, var, [], [], [], [])

        for varentry in self.cache_variables.values():
            for transentry in self.cache_entries:
                dist = transentry.tainted.distance(varentry.variable, True, 1)
                if dist!=1:
                    continue
                varentry.taints_entry.append(transentry)

                # print(varentry.variable.name(), '->', transentry.name)

            for varentry1 in self.cache_variables.values():
                dist = varentry1.variable.distance(varentry.variable, True, 1)
                if dist!=1:
                    continue
                varentry.taints_var.append(varentry1)
                varentry1.depends_var.append(varentry)

                # print(varentry.variable.name(), '->', varentry1.variable.name())

        for varhash in list(self.cache_variables.keys()):
            varentry = self.cache_variables.get(varhash, None)
            if varentry is None:
                continue

            self._remove_deadend_variable(varhash, varentry)

    def _remove_deadend_variable(self, varhash, varentry):
        # print('check', varentry.variable.name())
        if varentry.taints_var or varentry.taints_entry:
            return

        # print('  remove', varentry.variable.name())

        del self.cache_variables[varhash]
        upstream = varentry.depends_var

        for varentry1 in varentry.taints_var:
            varentry1.depends_var.remove(varentry)

        for varentry1 in upstream:
            varentry1.taints_var.remove(varentry)
            self._remove_deadend_variable(varentry1.variable.hash(), varentry1)

    def _list_do(self, lst, *args):
        for obj in lst:
            for fcn in args:
                fcn(obj)

    def entry_do(self, *args):
        return self._list_do(self.cache_entries, *args)

    def sink_do(self, *args):
        return self._list_do(self.cache_sinks, *args)

    def source_do(self, *args):
        return self._list_do(self.cache_sources, *args)

    def variable_do(self, *args):
        return self._list_do(self.cache_variables.values(), *args)

    def source_open_do(self, *args):
        return self._list_do(self.cache_sources_open, *args)

    def get_edges(self):
        edges=0
        for source in self.cache_sources:
            if source.sink:
                edges+=1
        return edges

    def get_mem_stats(self):
        nbytes, nelements = 0, 0
        for sink in self.cache_sinks:
            data = sink.data
            size = data.type.size()

            esize = N.finfo(data.buffer.typecode).bits//8

            nelements+=size
            nbytes+=size*esize

        return dict(bytes=nbytes, elements=nelements)

    def get_stats(self, fmt=None):
        stats = dict(
                nodes   = len(self.cache_entries),
                sources = len(self.cache_sources),
                sinks   = len(self.cache_sinks),
                edges   = self.get_edges()
                )

        stats.update(self.get_mem_stats())

        if fmt:
            return fmt.format(**stats)

        return stats

    def get_times(self, n):
        times = N.zeros(len(self.cache_entries))
        for i, entry in enumerate(self.cache_entries):
            res = self.benchmark(entry, n)
            times[i]=res

        return times

    def benchmark(self, entry, n):
        entry.evaluate()

        from gna.graph.timeit import timeit
        return timeit(entry.evaluate, n, lambda:None)/n
