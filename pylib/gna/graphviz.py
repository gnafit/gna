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
        if not isinstance(transformation, R.TransformationTypes.Handle):
            raise TypeError('GNADot argument should be of type TransformationDescriptor or TransformationTypes::Handle')

        self.graph=G.AGraph( directed=True, label=transformation.name(), **kwargs )
        self.register = set()
        self.walk_back( R.OpenHandle(transformation).getEntry() )
        self.write = self.graph.write

    def registered( self, *args, **kwargs ):
        id = uid( *args )
        if id in self.register:
            return True

        if kwargs.pop('register', True):
            self.register.add( id )
        return False

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
        else:
            labels['arrowhead']='empty'
        return {k:v for k, v in labels.items() if v}

    def get_subgraph(self, subname, label, graph=None):
        graph = graph or self.graph
        if not subname:
            return graph

        subname = 'cluster_'+subname
        graph = graph.get_subgraph(subname) or graph.add_subgraph(name=subname, label=label)

        return graph

    def get_graph(self, name):
        return self.graph
        subname = None
        label = None
        graph = self.graph

        if 'AD' in name:
            idx = name.find('AD')
            ad = name[idx:idx+4]
            subname=ad
            label=ad

            eh = 'EH'+ad[2]
            print( ad, eh )
            graph = self.get_subgraph(eh, eh, graph)
        elif 'EH' in name:
            idx = name.find('EH')
            eh = name[idx:idx+3]
            subname = eh
            label=eh
            print(eh )
        elif 'NL' in name:
            subname = 'NL'
            label='NL'

        return self.get_subgraph(subname, label, graph)

    def walk_back( self, entry ):
        if self.registered( entry ):
            return

        name = self.entryfmt.format(name=entry.name, label=entry.label)
        graph = self.get_graph(name)
        node = graph.add_node( uid(entry), label=name )
        self.walk_forward( entry, graph )

        for i, source in enumerate(entry.sources):
            assert source.materialized()
            sink = source.sink

            if self.registered( sink.entry, entry ):
                continue

            graph.add_edge( uid(sink.entry), uid(entry), **self.get_labels(None, sink, i, source))

            self.walk_back( sink.entry )

    def walk_forward( self, entry, graph=None ):
        graph = graph or self.graph
        for i, sink in enumerate( entry.sinks ):
            if sink.sources.size()==0:
                graph.add_node( uid(sink.entry)+' out', shape='point', label='out' )
                graph.add_edge( uid(sink.entry), uid(sink.entry)+' out', **self.get_labels(i, sink) )
                continue

            for j, source in enumerate(sink.sources):
                assert source.materialized()

                if self.registered( sink.entry, source.entry ):
                    continue
                graph.add_edge( uid(sink.entry), uid(source.entry), **self.get_labels(i, sink, None, source))

                self.walk_back( source.entry )

