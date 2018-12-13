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

    gdot = GNADot(obj)

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
        self.register = set()
        if not isinstance(transformation, (list, tuple)):
            transformation = [transformation]
        for t in transformation:
            if isinstance(t, R.TransformationTypes.OutputHandle):
                entry = R.OpenOutputHandle(t).getEntry()
            elif isinstance(t, R.TransformationTypes.Handle):
                entry = R.OpenHandle(t).getEntry()
            elif isinstance(t, R.SingleOutput):
                entry = R.OpenOutputHandle(t.single()).getEntry()
            else:
                raise TypeError('GNADot argument should be of type TransformationDescriptor/TransformationTypes::Handle/TransformationTypes::OutputHandle, got '+type(t).__name__)

            self.walk_back( entry )

        self.layout = self.graph.layout
        self.write = self.graph.write
        self.draw = self.graph.draw

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
        if graph is None:
            graph = self.graph

        if not subname:
            return graph

        subname = 'cluster_'+subname
        subgraph = graph.get_subgraph(subname)
        if subgraph is None:
            subgraph=graph.add_subgraph(name=subname, label=label)

        return subgraph

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

        """Add a node for current Entry"""
        node = graph.add_node( uid(entry), label=name )

        """For each sink of the Entry walk forward and build tree"""
        self.walk_forward( entry )

        """For each source of the Entry walk backward and build the tree"""
        for i, source in enumerate(entry.sources):
            if not source.sink:
                graph.add_node( uid(source)+' in', shape='point', label='in' )
                graph.add_edge( uid(source)+' in', uid(entry), **self.get_labels(i, source) )
                continue

            self.walk_back( source.sink.entry )

    def walk_forward( self, entry, graph=None ):
        if graph is None:
            graph = self.graph

        for i, sink in enumerate( entry.sinks ):
            if self.registered( sink ):
                continue

            if sink.sources.size()==0:
                """In case sink is not connected, draw empty output"""
                graph.add_node( uid(sink)+' out', shape='point', label='out' )
                graph.add_edge( uid(sink.entry), uid(sink)+' out', **self.get_labels(i, sink) )
                continue
            elif sink.sources.size()==1 or not self.joints:
                """In case there is only one connection draw it as is"""
                for j, source in enumerate(sink.sources):
                    graph.add_edge( uid(sink.entry), uid(source.entry), sametail=str(i), **self.get_labels(i, sink, None, source))
                    self.walk_back( source.entry )
            else:
                """In case there is more than one connections, merge them"""
                joint = graph.add_node( uid(sink), shape='none', width=0, height=0, penwidth=0, label='', xlabel=self.get_tail_label(None, sink) )
                graph.add_edge( uid(sink.entry), uid(sink), arrowhead='none', weight=0.5 )
                for j, source in enumerate(sink.sources):
                    graph.add_edge( uid(sink), uid(source.entry), **self.get_labels(i, None, None, source))
                    self.walk_back( source.entry )

