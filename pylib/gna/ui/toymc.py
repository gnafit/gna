from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('-a', '--add', nargs=2, action='append', default=[],
                            metavar=('NAME', 'PREDICTION'),
                            help='set data NAME to Asimov from prediction PREDICTION')
        parser.add_argument('-p', '--print', action='append', default=[],
                            metavar='NAME',
                            help='print data NAME')

    def init(self):
        for name, predname in self.opts.add:
            prediction = self.env.predictions[predname]
            buf = np.frombuffer(prediction.data(), count=prediction.size())
            data = ROOT.Points(np.random.normal(buf, buf**0.5))
            self.env.adddata(name, data)
            print 'Asimov', name, 'for', predname

        for name in getattr(self.opts, 'print'):
            data = self.env.data[name]
            print name
            print np.frombuffer(data.data(), count=data.size())
