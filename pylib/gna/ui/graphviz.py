from __future__ import print_function
from gna.ui import basecmd, append_typed, qualified
from gna.env import env, PartNotFoundError
import pygraphviz as G
import ROOT as R

from gna.graphviz import GNADot


class cmd(basecmd):
    undefined = dict()

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
        parser.add_argument('-J', '--no-joints', action='store_false', dest='joints', help='disable joints')
        parser.add_argument('--subgraph', action='store_true', help='enable subgraphs')
        parser.add_argument('-s', '--splines', help='splines option [dot]')
        parser.add_argument('-o', '--output', nargs='+', default=[], dest='outputs', help='output dot/pdf/png file')
        parser.add_argument('-O', '--stdout', action='store_true', help='output to stdout')
        parser.add_argument('-E', '--stderr', action='store_true', help='output to stderr')
        parser.add_argument('-n', '--namespace', '--ns', default=cls.undefined, nargs='?', help='use <namespace> to read parameters', metavar='namespace')
        parser.add_argument('--option', nargs=2, action='append', dest='options', default=[], help='AGraph kwargs key value pair')

    def init(self):
        head = self.opts.plot[0]

        kwargs = dict(self.opts.options, joints=self.opts.joints)
        kwargs.setdefault('rankdir', 'LR')
        kwargs['subgraph']=self.opts.subgraph
        if self.opts.splines:
            kwargs['splines']=self.opts.splines

        if self.opts.namespace is not self.undefined:
            kwargs['namespace']=env.globalns(self.opts.namespace)

        graph = GNADot( head, **kwargs )


        for output in self.opts.outputs:
            print( 'Write graph to:', output )

            if output.endswith('.dot'):
                graph.write(output)
            else:
                graph.layout(prog='dot')
                graph.draw(output)

        if self.opts.stdout:
            graph.write()

        if self.opts.stderr:
            import sys
            graph.write( sys.stderr )

