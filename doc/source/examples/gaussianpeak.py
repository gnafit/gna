from gna.ui import basecmd
from gna.env import env
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--Emin', default=0, type=float)
        parser.add_argument('--Emax', default=5, type=float)
        parser.add_argument('--nbins', default=20, type=int)
        parser.add_argument('--order', default=5)

    def init(self):
        ns = env.ns(self.opts.name)
        ns.reqparameter('BackgroundRate', central=1, sigma=0.1)
        ns.reqparameter('Mu', central=0, sigma=1)
        ns.reqparameter('E0', central=1, sigma=0.05)
        ns.reqparameter('Width', central=0.2, sigma=0.005)
        with ns:
            model = ROOT.GaussianPeakWithBackground()

        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1)
        orders = np.array([self.opts.order]*(len(edges)-1), dtype=int)

        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        model.rate.E(integrator.points.x)
        hist = ROOT.GaussLegendreHist(integrator)
        hist.hist.f(model.rate.rate)

        ns.addobservable('spectrum', hist.hist)
