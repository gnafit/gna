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
        graph=G.AGraph( directed=True, label=head.name() )

        walk_back( head, graph, register=set() )

        if self.opts.output:
            print( 'Write graph to:', self.opts.output )
            graph.write( self.opts.output )

        if self.opts.stdout:
            graph.write()

        if self.opts.stderr:
            import sys
            graph.write( sys.stderr )

def walk_back( obj, graph, register ):
    entry_hash = obj.__hash__()

    print( obj.name(), entry_hash )
    if entry_hash in register:
        print( '   skip, been there' )
        return
    register.add(entry_hash)

    node = graph.add_node( entry_hash, label=obj.name() )

    for iname, input in obj.inputs.iteritems():
        parent = input.parent()
        import IPython
        IPython.embed()
        # walk_back( parent, graph, register )

        graph.add_edge( parent.hash(), hash, taillabel=iname )

