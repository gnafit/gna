# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from gna.env import env
import pygraphviz as G
import ROOT as R
from collections import OrderedDict
from .configurator import NestedDict

import re
pattern = re.compile('^.*::([^:<]+)(T<[^>]*>)*$')

def uid( obj1, obj2=None ):
    if obj2:
        res = '%s -> %s'%( uid( obj1), uid( obj2 ) )
    else:
        res = obj1.__repr__().replace('*', '')
    return res

def savegraph(obj, fname, *args, **kwargs):
    if not fname:
        return

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
    layers = 'variable:transformation'
    make_subgraphs = True
    def __init__(self, transformation, **kwargs):
        kwargs.setdefault('fontsize', 10)
        kwargs.setdefault('labelfontsize', 10)
        kwargs.setdefault('rankdir', 'LR')
        self.make_subgraphs = kwargs.get('subgraph', False)
        self.joints = kwargs.pop('joints', False)
        ns = kwargs.pop('namespace', None)

        self.subgraphs = dict()

        self.graph=G.AGraph(directed=True, strict=False, layers=self.layers, **kwargs)
        self.layout = self.graph.layout
        self.write = self.graph.write
        self.draw = self.graph.draw

        self._entry_uids = OrderedDict()

        from gna.graph.walk import GraphWalker
        self.walker = GraphWalker(transformation, namespace=ns)
        self.style=TreeStyle(self.walker)

        self.walker.entry_do(self._action_entry)
        self.walker.source_open_do(self._action_source_open)

        self._process_variables()

    def _get_subgraph(self, entry):
        if not self.make_subgraphs:
            return self.graph

        if not 'subgraph' in entry.attrs:
            return self.graph

        name = entry.attrs['subgraph']

        subgraph = self.subgraphs.get(name)
        if subgraph is not None:
            return subgraph

        subgraph=self.subgraphs[name]=self.graph.add_subgraph(name='cluster_'+name, label=name)

        return subgraph

    def _process_variables(self):
        self.vargraph=self.graph
        self.walker.variable_do(self._action_variable)

    def entry_uid(self, entry, suffix=None):
        hash=entry.hash()
        uid = self._entry_uids.get(hash, None)
        if uid:
            return uid

        uid = ('entry', '%04i'%(len(self._entry_uids)))
        if suffix:
            uid+=suffix,
        uid='_'.join(uid)
        self._entry_uids[hash]=uid

        return uid

    def _action_entry(self, entry):
        graph = self._get_subgraph(entry)

        node = graph.add_node( self.entry_uid(entry), **self.style.node_attrs(entry) )
        nsinks = entry.sinks.size()
        for i, sink in enumerate(entry.sinks):
            self._action_sink(sink, i, nsinks, graph=graph)

    def _action_variable(self, varentry):
        var=varentry.variable
        varid = self.entry_uid(var)
        node = self.vargraph.add_node(varid, **self.style.node_attrs_var(varentry))

        for entry in varentry.taints_entry:
            # print('plot entry', varentry.variable.name(), '->', entry.name, self.entry_uid(entry))
            self.vargraph.add_edge(varid, self.entry_uid(entry), **self.style.edge_attrs_var(var, False))

        for varentry1 in varentry.taints_var:
            varid1 = self.entry_uid(varentry1.variable)
            # print('plot var', varentry.variable.name(), '->', varentry1.variable.name(), varid1)
            self.vargraph.add_edge(varid, varid1, **self.style.edge_attrs_var(var, True))

    def _action_source_open(self, source, i=0):
        sourceuid = self.entry_uid(source, 'source')
        self.graph.add_node( sourceuid, shape='point', label='in' )
        self.graph.add_edge( sourceuid, self.entry_uid(source.entry), **self.style.edge_attrs(i, source) )

    def _action_sink(self, sink, i=0, nsinks=0, graph=None):
        graph = graph or self.graph
        if sink.sources.size()==0:
            """In case sink is not connected, draw empty output"""
            sinkuid=self.entry_uid(sink, 'sink')
            graph.add_node( sinkuid, shape='point', label='out' )
            graph.add_edge( self.entry_uid(sink.entry), sinkuid, **self.style.edge_attrs(i, sink) )
        elif sink.sources.size()==1 or not self.joints:
            """In case there is only one connection draw it as is"""
            sinkuid = self.entry_uid(sink.entry)
            sametail=str(i) if nsinks<5 else None
            for j, source in enumerate(sink.sources):
                self.graph.add_edge( sinkuid, self.entry_uid(source.entry), sametail=sametail, **self.style.edge_attrs(i, sink, None, source))
        else:
            """In case there is more than one connections, merge them"""
            jointuid = self.entry_uid(sink, 'joint')
            joint = graph.add_node( jointuid, shape='none', width=0, height=0, penwidth=0, label='', xlabel=self.style.tail_label(None, sink) )

            sstyle=self.style.edge_attrs(i, sink, None, None)
            sstyle['arrowhead']='none'
            self.graph.add_edge( self.entry_uid(sink.entry), jointuid, weight=0.5, **sstyle )
            for j, source in enumerate(sink.sources):
                self.graph.add_edge( jointuid, self.entry_uid(source.entry), **self.style.edge_attrs(i, sink, None, source))

class TreeStyle(object):
    markhead, marktail = True, True
    # headfmt = '{index:d}: {label}'
    headfmt = '{label}'
    headfmt_noi = '{label}'
    # tailfmt = '{index:d}: {label}'
    tailfmt = '{label}'
    tailfmt_noi = '{label}'
    entryfmt = '{label}'

    gpucolor = 'limegreen'
    varcolor = 'cornflowerblue'
    staticcolor = 'azure3'
    gpupenwidth = 4

    def __init__(self, walker):
        self.walker=walker

        self.entry_features=dict()

    def build_features(self, entry):
        attrs=dict(entry.attrs)
        objectname = attrs['_object']
        entryname = entry.name
        funcname = entry.funcname

        if '::' in objectname:
            objectname = pattern.match(objectname).groups()[0]

        features = NestedDict(static=False, gpu=False, label=attrs['_label'], frozen=entry.tainted.frozen())

        def getdim(sink, offset=0):
            if not sink.materialized():
                return '?',

            return tuple('%i'%(d+offset) for d in sink.data.type.shape)

        dim=None

        marks = {
                'Sum':               '+',
                'MultiSum':          '+',
                'SumBroadcast':      '+',
                'WeightedSum':       '+w',
                'Product':           '*',
                'Points':            'a',
                'View':              'v',
                'Histogram':         'h',
                'Histogram2d':       'h²',
                'Rebin':             'r',
                'Concat':            '..',
                'FillLike':          'c',
                'HistSmearSparse':   '@',
                'HistSmear':         '@',
                'MatrixProduct':     '@',
                'Snapshot':          r'\|o\|',
                'MatrixProductDVDt': '@@t',
                'InSegment':         '∈'
                }
        if objectname in marks:
            mark = marks.get(objectname)
            dim = getdim(entry.sinks.back())
        else:
            mark = None

        if objectname in ('Points', 'Histogram', 'Histogram2d', 'FillLike',):
            features.static=True

        npars = 0
        if objectname in ('WeightedSum'):
            npars=entry.sources.size()


        if objectname.startswith('Integrator'):
            if entry.name == 'points':
                mark='x'
                features.static=True
                if objectname.startswith('Integrator21'):
                    dim = getdim(entry.sinks[3])
                elif objectname.startswith('Integrator2'):
                    dim = getdim(entry.sinks[4])
                else:
                    dim = getdim(entry.sinks[0])
            else:
                mark='i'
                dim = getdim(entry.sinks[0])
        elif objectname.startswith('Interp'):
            mark='~'
            dim = getdim(entry.sinks.back())
        elif objectname in ('InSegment',):
            dim1 = 'x'.join(getdim(entry.sinks[0]))
            dim2 = 'x'.join(getdim(entry.sinks[1], 1))
            dim = ']∈['.join((dim1, dim2)),

        if entry.funcname=='gpu':
            features.gpu=True

        features.dim=dim
        features.npars=npars
        features.mark=mark
        self.entry_features[entry.hash()]=features
        return features

    def get_features(self, entry):
        features = self.entry_features.get(entry.hash(), None)
        if features is None:
            return self.build_features(entry)

        return features

    def node_attrs_var(self, varentry):
        ret=dict(shape='octagon', style='rounded', fontsize=10, color=self.varcolor, layer='variable')
        variable=varentry.variable

        label=variable.name()
        value = str(variable.value())
        label+='='+value

        size=variable.values().size()
        if size>1:
            label='%s [%i]'%(label, size)

        label='%s\n%s'%(label, varentry.fullname)

        ret['label']=label
        return ret

    def node_attrs(self, entry):
        ret=dict(shape='Mrecord', layer='transformation')
        features=self.get_features(entry)

        styles=()
        label=()
        color=None

        if features.frozen:
            styles+='dashed',

        if features.gpu:
            ret['color']=self.gpucolor
            styles+='bold',
        if features.static:
            ret['color']=self.staticcolor

        label=self.entryfmt.format(name=entry.name, label=features['label'])

        mark=features.mark
        dim=features.dim
        npars=features.npars
        marks = ()
        if mark:
            marks+=mark,
        if npars:
            marks+='(%i)'%npars,
            marks='{%s}'%('|'.join(marks)),
        if dim:
            marks+='[%s]'%('x'.join(dim)),
            marks='{%s}'%('|'.join(marks)),
        if marks:
            marks+=label,
            marks='{%s}'%('|'.join(marks)),
        else:
            marks=label,

        ret['label'] = marks[0]
        ret['style'] = ','.join(styles)
        return ret

    def head_label(self, i, obj):
        attrs=dict(obj.attrs)
        if not self.markhead:
            return None
        if isinstance(obj, str):
            return obj

        if i is None:
            return self.headfmt_noi.format(name=obj.name, label=attrs.get('_label', ''))

        return self.headfmt.format(index=i, name=obj.name, label=attrs.get('_label', ''))

    def tail_label(self, i, obj):
        attrs=dict(obj.attrs)
        if not self.marktail:
            return None
        if isinstance(obj, str):
            return obj

        if i is None:
            return self.tailfmt_noi.format(name=obj.name, label=attrs.get('_label', ''))

        return self.tailfmt.format(index=i, name=obj.name, label=attrs.get('_label', ''))

    def edge_attrs(self, isink, sink, isource=None, source=None):
        attrs = dict(layer='transformation')
        style=()
        if sink:
            taillabel=self.tail_label(isink, sink)
            if taillabel and source:
                attrs['xlabel']=taillabel

            sinkfeatures = self.get_features(sink.entry)
            if sinkfeatures.static:
                attrs['color']=self.staticcolor

            gpu1 = sinkfeatures.gpu
        else:
            gpu1 = False

        if source:
            attrs['headlabel']=self.head_label(isource, source)
            sourcefeatures = self.get_features(source.entry)
            gpu2 = sourcefeatures.gpu
            if sourcefeatures.frozen:
                attrs['arrowhead']='tee'
                style+='dashed',
        else:
            attrs['arrowhead']='empty'
            gpu2 = False

        if gpu1 or gpu2:
            attrs['color']    =self.gpucolor

        if gpu1^gpu2:
            style+='tapered',
            attrs['penwidth'] =self.gpupenwidth
            attrs['arrowhead']='none'

        if gpu1 and gpu2:
            attrs['dir']      ='forward'
        elif gpu1:
            attrs['dir']      ='forward'
        elif gpu2:
            attrs['dir']      ='back'
            attrs['arrowtail']='none'

        attrs['style']=','.join(style)
        return {k:v for k, v in attrs.items() if v}

    def edge_attrs_var(self, var, tovar=False):
        ret=dict(color=self.varcolor, layer='variable')
        if tovar:
            ret['weight']=0.5
        else:
            ret['weight']=1.0

        return ret
