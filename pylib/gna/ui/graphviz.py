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

        graph = GNADot( head )

        if self.opts.output:
            print( 'Write graph to:', self.opts.output )
            graph.write( self.opts.output )

        if self.opts.stdout:
            graph.write()

        if self.opts.stderr:
            import sys
            graph.write( sys.stderr )

def uid( obj1, obj2=None ):
    if obj2:
        return '%s -> %s'%( uid( obj1), uid( obj2 ) )

    return obj1.__repr__().replace('*', '')

class GNADot(object):
    def __init__(self, transformation):
        self.graph=G.AGraph( directed=True, label=transformation.name() )
        self.register = set()
        self.walk_back( transformation.entry() )
        self.write = self.graph.write

    def registered( self, *args ):
        id = uid( *args )
        if id in self.register:
            return True

        self.register.add( id )
        return False

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

            self.graph.add_edge( uid(sink.entry), uid(entry), headlabel='%i: %s'%(i, source.name), taillabel=sink.name )

            self.walk_back( sink.entry )

    def walk_forward( self, entry ):
        for i, sink in enumerate( entry.sinks ):
            if sink.sources.size()==0:
                self.graph.add_node( uid(sink.entry)+' out', shape='point', label='out' )
                self.graph.add_edge( uid(sink.entry), uid(sink.entry)+' out', taillabel='%i: %s'%(i, sink.name), arrowhead='empty' )
                continue

            for source in sink.sources:
                assert source.materialized()

                if self.registered( sink.entry, source.entry ):
                    continue
                self.graph.add_edge( uid(sink.entry), uid(source.entry), headlabel='%s'%(source.name), taillabel='%i: %s'%(i, sink.name) )

                if not self.registered( source.entry ):
                    node = self.graph.add_node( uid(source.entry), label=source.entry.name )
                    self.walk_forward( source.entry )

