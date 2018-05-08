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
    markhead, marktail = True, False
    # headfmt = '{index:d}: {name}'
    headfmt = '{name}'
    headfmt_noi = '{name}'
    # tailfmt = '{index:d}: {name}'
    tailfmt = '{name}'
    tailfmt_noi = '{name}'
    def __init__(self, transformation):
        if not isinstance(transformation, R.TransformationTypes.Handle):
            raise TypeError('GNADot argument should be of type TransformationDescriptor or TransformationTypes::Handle')

        self.graph=G.AGraph( directed=True, label=transformation.name() )
        self.register = set()
        self.walk_back( R.TransformationTypes.OpenHandle(transformation).getEntry() )
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
            return self.headfmt_noi.format(name=obj.name)

        return self.headfmt.format(index=i, name=obj.name)

    def get_tail_label(self, i, obj):
        if not self.marktail:
            return None
        if isinstance(obj, basestring):
            return obj

        if i is None:
            return self.tailfmt_noi.format(name=obj.name)

        return self.tailfmt.format(index=i, name=obj.name)

    def get_labels(self, isink, sink, isource=None, source=None):
        labels = {}
        if sink:
            labels['taillabel']=self.get_tail_label(isink, sink)
        if source:
            labels['headlabel']=self.get_head_label(isource, source)
        else:
            labels['arrowhead']='empty'
        return {k:v for k, v in labels.items() if v}

    def walk_back( self, entry ):
        if self.registered( entry ):
            return

        node = self.graph.add_node( uid(entry), label=entry.name )
        self.walk_forward( entry )

        for i, source in enumerate(entry.sources):
            assert source.materialized()
            sink = source.sink

            if self.registered( sink.entry, entry ):
                continue

            self.graph.add_edge( uid(sink.entry), uid(entry), **self.get_labels(None, sink, i, source))

            self.walk_back( sink.entry )

    def walk_forward( self, entry ):
        for i, sink in enumerate( entry.sinks ):
            if sink.sources.size()==0:
                self.graph.add_node( uid(sink.entry)+' out', shape='point', label='out' )
                self.graph.add_edge( uid(sink.entry), uid(sink.entry)+' out', **self.get_labels(i, sink) )
                continue

            for j, source in enumerate(sink.sources):
                assert source.materialized()

                if self.registered( sink.entry, source.entry ):
                    continue
                self.graph.add_edge( uid(sink.entry), uid(source.entry), **self.get_labels(i, sink, None, source))

                self.walk_back( source.entry )

