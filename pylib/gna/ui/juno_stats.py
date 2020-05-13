"""Print JUNO related stats and plot default figures, latex friendly"""
from __future__ import print_function
from gna.ui import basecmd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('exp', help='JUNO exp instance')

    def init(self):
        try:
            self.exp = self.env.parts.exp[self.opts.exp]
        except Exception:
            print('Unable to retrieve exp '+self.opts.exp)

        self.namespace = self.exp.namespace
        self.context   = self.exp.context
        self.outputs   = self.exp.context.outputs
        self.observation = self.outputs.observation.AD1

        self.init_variables()

    def init_variables(self):
        ns = self.namespace

        import IPython; IPython.embed()

