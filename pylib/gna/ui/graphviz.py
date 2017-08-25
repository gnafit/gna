from __future__ import print_function
from gna.ui import basecmd, append_typed, qualified
from gna.env import env
import pygraphviz as G

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            nspath, name = path.split('/')
            try:
                return env.ns(nspath).observables[name]
            except KeyError:
                raise PartNotFoundError("observable", path)

        parser.add_argument('plot', default=[],
                            metavar=('DATA',),
                            action=append_typed(observable))
        parser.add_argument('-o', '--output', help='output .dot file')
        parser.add_argument('-O', '--stdout', action='store_true', help='output to stdout')
        parser.add_argument('-E', '--stderr', action='store_true', help='output to stderr')

    def init(self):
        head = self.opts.plot[0]

        graph = plot_transformation( head )

        if self.opts.output:
            print( 'Write graph to:', self.opts.output )
            graph.write( self.opts.output )

        if self.opts.stdout:
            graph.write()

        if self.opts.stderr:
            import sys
            graph.write( sys.stderr )

def plot_transformation( obj ):
    graph=G.AGraph( directed=True, label=obj.name() )
    walk_back( obj.entry(), graph, set() )
    return graph

def uid( obj1, obj2=None ):
    if obj2:
        return '%s -> %s'%( uid( obj1), uid( obj2 ) )

    return obj1.__repr__().replace('*', '')

def registered( id, register ):
    if id in register:
        return True

    register.add( id )
    return False

def walk_back( entry, graph, register ):
    id = uid( entry )
    print( entry.name, id )
    if registered( id, register ):
        return

    node = graph.add_node( uid(entry), label=entry.name )
    walk_forward( entry, graph, register )

    for i, source in enumerate(entry.sources):
        assert source.materialized()
        sink = source.sink

        if registered( uid( sink.entry, entry ), register ):
            continue
        graph.add_edge( uid(sink.entry), uid(entry), headlabel='%i: %s'%(i, source.name), taillabel=sink.name )

        walk_back( sink.entry, graph, register )

def walk_forward( entry, graph, register ):
    for i, sink in enumerate( entry.sinks ):
        if sink.sources.size()==0:
            graph.add_node( uid(sink.entry)+' out', shape='point', label='out' )
            graph.add_edge( uid(sink.entry), uid(sink.entry)+' out', taillabel='%i: %s'%(i, sink.name), arrowhead='empty' )
            continue

        for source in sink.sources:
            assert source.materialized()

            if registered( uid( sink.entry, source.entry ), register ):
                continue
            graph.add_edge( uid(sink.entry), uid(source.entry), headlabel='%s'%(source.name), taillabel='%i: %s'%(i, sink.name) )

