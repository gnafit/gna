from gna.ui import basecmd, append_typed
import ROOT
import numpy as np
from gna.env import PartNotFoundError

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        def observablespec(spec):
            ret = []
            for path in spec.split('+'):
                nspath, name = path.split('/')
                try:
                    ret.append(env.ns(nspath).observables[name])
                except KeyError:
                    raise PartNotFoundError("observable", path)
            return ret

        parser.add_argument('-l', '--list-observables', action='store_true',
                            help='display all available observables')
        parser.add_argument('-a', '--add', nargs=2, default=[],
                            metavar=('NAME', 'OBSSPEC'),
                            action=append_typed(str, observablespec),
                            help='add prediction NAME for observables specified by OBSSPEC (o1[+o2+[...]])')
        parser.add_argument('-p', '--print', action='append', default=[],
                            metavar='NAME',
                            type=env.parts.prediction,
                            help='print prediction NAME')

    def init(self):
        if self.opts.list_observables:
            for ns in self.env.iternstree():
                for name, prediction in ns.observables.iteritems():
                    path = '/'.join([ns.path, name])
                    print 'Observable', path

        for name, observables in self.opts.add:
            prediction = ROOT.Prediction()
            for obs in observables:
                prediction.append(obs)
            self.env.parts.prediction[name] = prediction
            print 'Prediction', name, 'size:', prediction.size()

        for prediction in getattr(self.opts, 'print'):
            print np.frombuffer(prediction.data(), count=prediction.size())

