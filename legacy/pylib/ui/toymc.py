from gna.ui import basecmd, append_typed
import ROOT
import gna.constructors as C
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name')
        parser.add_argument('analysis', type=env.parts.analysis)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-a', '--add', nargs=2, default=[],
                            metavar=('NAME', 'PREDICTION'),
                            action=append_typed(str, env.parts.prediction),
                            help='set data NAME to Asimov from prediction PREDICTION')
        parser.add_argument('-p', '--print', action='append', default=[],
                            metavar='NAME',
                            type=env.parts.data,
                            help='print data NAME')

    def init(self):
        for name, prediction in self.opts.add:
            buf = prediction.data()
            # data = C.Points(np.random.normal(buf, buf**0.5))
            data = C.Points(buf)
            self.env.parts.data[name] = data
            print 'Asimov', name

        for data in getattr(self.opts, 'print'):
            print name
            print data.data()
