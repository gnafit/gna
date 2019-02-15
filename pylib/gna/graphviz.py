from __future__ import print_function
from gna.env import env
import pygraphviz as G
import ROOT as R

def uid( obj1, obj2=None ):
    if obj2:
        res = '%s -> %s'%( uid( obj1), uid( obj2 ) )
    else:
        res = obj1.__repr__().replace('*', '')
    return res

def savegraph(obj, fname, *args, **kwargs):
    verbose = kwargs.pop('verbose', True)

    gdot = GNADot(obj, *args, **kwargs)

    if verbose:
        print('Write output file:', fname)

    if fname.endswith('.dot'):
        gdot.write(fname)
    else:
        gdot.layout(prog='dot')
        gdot.draw(fname)

class GNADot(object):
    markhead, marktail = True, True
    # headfmt = '{index:d}: {label}'
    headfmt = '{label}'
    headfmt_noi = '{label}'
    # tailfmt = '{index:d}: {label}'
    tailfmt = '{label}'
    tailfmt_noi = '{label}'
    entryfmt = '{label}'

    def __init__(self, transformation, **kwargs):
        kwargs.setdefault('fontsize', 10)
        kwargs.setdefault('labelfontsize', 10)
        self.joints = kwargs.pop('joints', False)

        self.graph=G.AGraph( directed=True, strict=False, **kwargs )
        self.layout = self.graph.layout
        self.write = self.graph.write
        self.draw = self.graph.draw

        from gna.graph.walk import GraphWalker
        self.walker = GraphWalker(transformation)

        self.walker.entry_do(self._action_entry)
        self.walker.source_open_do(self._action_source_open)

    def _get_graph(self, *args):
        return self.graph

    def _action_entry(self, entry):
        name = self.entryfmt.format(name=entry.name, label=entry.label)
        graph = self._get_graph(name)

        node = graph.add_node( uid(entry), label=name )
        for i, sink in enumerate(entry.sinks):
            self._action_sink(sink, i)

    def _action_source_open(self, source, i=0):
        graph.add_node( uid(source)+' in', shape='point', label='in' )
        graph.add_edge( uid(source)+' in', uid(source.entry), **self.get_labels(i, source) )

    def _action_sink(self, sink, i=0):
        if sink.sources.size()==0:
            """In case sink is not connected, draw empty output"""
            self.graph.add_node( uid(sink)+' out', shape='point', label='out' )
            self.graph.add_edge( uid(sink.entry), uid(sink)+' out', **self.get_labels(i, sink) )
        elif sink.sources.size()==1 or not self.joints:
            """In case there is only one connection draw it as is"""
            for j, source in enumerate(sink.sources):
                self.graph.add_edge( uid(sink.entry), uid(source.entry), sametail=str(i), **self.get_labels(i, sink, None, source))
        else:
            """In case there is more than one connections, merge them"""
            joint = self.graph.add_node( uid(sink), shape='none', width=0, height=0, penwidth=0, label='', xlabel=self.get_tail_label(None, sink) )
            self.graph.add_edge( uid(sink.entry), uid(sink), arrowhead='none', weight=0.5 )
            for j, source in enumerate(sink.sources):
                self.graph.add_edge( uid(sink), uid(source.entry), **self.get_labels(i, None, None, source))

    def get_head_label(self, i, obj):
        if not self.markhead:
            return None
        if isinstance(obj, basestring):
            return obj

        if i is None:
            return self.headfmt_noi.format(name=obj.name, label=obj.label)

        return self.headfmt.format(index=i, name=obj.name, label=obj.label)

    def get_tail_label(self, i, obj):
        if not self.marktail:
            return None
        if isinstance(obj, basestring):
            return obj

        if i is None:
            return self.tailfmt_noi.format(name=obj.name, label=obj.label)

        return self.tailfmt.format(index=i, name=obj.name, label=obj.label)

    def get_labels(self, isink, sink, isource=None, source=None):
        labels = {}
        if sink:
            labels['taillabel']=self.get_tail_label(isink, sink)
        if source:
            labels['headlabel']=self.get_head_label(isource, source)
            if source.entry.tainted.frozen():
                labels['style']='dashed'
        else:
            labels['arrowhead']='empty'
        return {k:v for k, v in labels.items() if v}

