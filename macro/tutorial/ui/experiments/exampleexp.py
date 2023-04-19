from gna.exp import baseexp
from gna import constructors as C
import ROOT as R
import numpy as np

class exp(baseexp):
    def __init__(self, namespace, opts):
        baseexp.__init__(self, namespace, opts)

    def init(self):
        self.build()
        self.register()

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument('-r', '--range',  default=(0.0, 10.0), type=float, nargs=2, help='Energy range')
        parser.add_argument('-N', '--points', default=200, type=int, help='Number of points')
        parser.add_argument('-n', '--name',   default='peak', help='observable name')

    def build(self):
        locns = self.namespace
        for name in ['BackgroundRate', 'Mu', 'E0', 'Width']:
            if not name in locns:
                locns.printparameters(labels=True)
                raise Exception('Paramter {name} not found in namespace'.format(name=name))

        self.points = C.Points(np.linspace(*self.opts.range, num=self.opts.points))
        self.model = C.GaussianPeakWithBackground(labels='Gaussian peak')
        self.points.points.points >> self.model.rate.E

    def register(self):
        self.namespace.addobservable('x', self.points.points.points)
        self.namespace.addobservable(self.opts.name, self.model.rate.rate)
