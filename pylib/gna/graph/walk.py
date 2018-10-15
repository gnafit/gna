# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
from collections import deque

class GraphWalker(object):
    def __init__(self, *args):
        self._entry_points = []
        for arg in args:
            self._add_entry_point(arg)

        self.build_cache()

    def _add_entry_point(self, arg):
        if isinstance(arg, R.TransformationTypes.OutputHandle):
            entry = R.OpenOutputHandle(arg).getEntry()
        elif isinstance(arg, R.TransformationTypes.Handle):
            entry = R.OpenHandle(arg).getEntry()
        else:
            raise TypeError('Unsupported argument type '+type(arg).__name__)

        self._entry_points.append(entry)

    def _propagate_forward(self, entry, queue):
        for sink in entry.sinks:
            for source in sink.sources:
                other = source.entry
                if other in self.cache_entries:
                    continue

                queue.append(other)

    def _propagate_backward(self, entry, queue):
        for source in entry.sources:
            entry=source.sink.entry
            if entry in self.cache_entries:
                continue

            queue.append(entry)

    def build_cache(self):
        self.cache_entries=[]
        self.cache_sources=[]
        self.cache_sinks=[]

        queue=deque(self._entry_points)
        while queue:
            entry = queue.popleft()

            self.cache_sources.extend((inp for inp in entry.sources if not inp in self.cache_sources))
            self.cache_sinks.extend(  (inp for inp in entry.sinks   if not inp in self.cache_sinks))
            self.cache_entries.append(entry)

            self._propagate_forward(entry, queue)
            self._propagate_backward(entry, queue)

    def entry_do(self, *args):
        for entry in self.cache_entries:
            for fcn in args:
                fcn(entry)

