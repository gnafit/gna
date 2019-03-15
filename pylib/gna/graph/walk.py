# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
from collections import deque
import numpy as N
import types

class GraphWalker(object):
    def __init__(self, *args):
        self._entry_points = []
        for arg in args:
            self._add_entry_point(arg)

        self._build_cache()

    def _add_entry_point(self, arg):
        OutputHandle = R.TransformationTypes.OutputHandleT('double')
        Handle = R.TransformationTypes.HandleT('double', 'double')
        SingleOutput = R.SingleOutputT('double')

        if isinstance(arg, (types.GeneratorType)):
            arg = list(arg)
        if not isinstance(arg, (list, tuple)):
            arg = [arg]

        for t in arg:
            if isinstance(t, OutputHandle):
                entry = R.OpenOutputHandleT('double','double')(t).getEntry()
            elif isinstance(t, Handle):
                entry = R.OpenHandleT('double','double')(t).getEntry()
            elif isinstance(t, SingleOutput):
                entry = R.OpenOutputHandleT('double','double')(t.single()).getEntry()
            else:
                # raise TypeError('GNADot argument should be of type TransformationDescriptor/TransformationTypes::Handle/TransformationTypes::OutputHandle, got '+type(t).__name__)
                raise TypeError('Unsupported argument type '+type(arg).__name__)

        self._entry_points.append(entry)

    def _propagate_forward(self, entry, queue):
        for sink in entry.sinks:
            for source in sink.sources:
                other = source.entry
                if other in queue or other in self.cache_entries:
                    continue

                queue.append(other)

            if sink.sources.size()==0:
                self.cache_sinks_open.append(sink)

    def _propagate_backward(self, entry, queue):
        for source in entry.sources:
            sink = source.sink

            if sink:
                other=sink.entry
                if other in queue or other in self.cache_entries:
                    continue

                queue.append(other)
            else:
                self.cache_sources_open.append(source)

    def _build_cache(self):
        self.cache_entries=[]
        self.cache_sources=[]
        self.cache_sinks=[]

        self.cache_sources_open=[]
        self.cache_sinks_open=[]

        queue=deque(self._entry_points)
        while queue:
            entry = queue.popleft()

            self.cache_entries.append(entry)
            self.cache_sources.extend((inp for inp in entry.sources if not inp in self.cache_sources))
            self.cache_sinks.extend(  (inp for inp in entry.sinks   if not inp in self.cache_sinks))

            self._propagate_forward(entry, queue)
            self._propagate_backward(entry, queue)

        self._build_cache_variables(self)

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
