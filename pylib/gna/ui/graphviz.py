from __future__ import print_function
from gna.ui import basecmd, append_typed, qualified
from gna.env import env, PartNotFoundError
import pygraphviz as G
import ROOT as R

from gna.graphviz import GNADot

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

