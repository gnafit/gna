"""Make a ToyMC based on an output"""

from gna.ui import basecmd
from gna import constructors as C

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, _):
        parser.add_argument('-r', '--root', default='spectra', help='root location for read/store actions')
        parser.add_argument('fcn', choices=('sqrt', 'sin', 'cos', 'exp'), help='function name')
        parser.add_argument('name_in', help='input observable name')
        parser.add_argument('name_out', help='observable name (output)')
        parser.add_argument('-l', '--label', help='node label')

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    def init(self):
        output = self.env.future[self.opts.root, self.opts.name_in]

        if not output:
            raise Exception('Invalid or missing output: {}'.format(self.opts.name_in))

        functions = { 'sin': C.Sin, 'cos': C.Cos, 'exp': C.Exp, 'sqrt': C.Sqrt }
        self.cls = functions[self.opts.fcn]
        self.instance = self.cls(output)

        trans = self.instance.transformations[0]
        trans.setLabel(f'{self.opts.fcn}' if self.opts.label is None else self.opts.label)
        self.env.future[self.opts.root, self.opts.name_out] = trans.single()

