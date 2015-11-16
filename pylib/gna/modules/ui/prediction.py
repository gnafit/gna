from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('-l', '--list-observables', action='store_true',
                            help='display all available observables')
        parser.add_argument('-a', '--add', nargs=2, action='append', default=[],
                            metavar=('NAME', 'OBSSPEC'),
                            help='add prediction NAME for observables specified by OBSSPEC (o1[+o2+[...]])')
        parser.add_argument('-p', '--print', action='append', default=[],
                            metavar='NAME',
                            help='print prediction NAME')

    def init(self):
        if self.opts.list_observables:
            for path in self.env.observables.iterpaths():
                print 'Observable', path

        for name, obsspec in self.opts.add:
            prediction = ROOT.Prediction()
            for obs in self.env.observables.fromspec(obsspec):
                prediction.append(obs)
            self.env.addprediction(name, prediction)
            print 'Prediction', name, 'size:', prediction.size()

        for name in getattr(self.opts, 'print'):
            prediction = self.env.predictions[name]
            print name
            print np.frombuffer(prediction.data(), count=prediction.size())

