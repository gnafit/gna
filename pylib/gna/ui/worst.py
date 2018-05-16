from gna.ui import basecmd
from gna.env import env
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--Emin', default=1.0, type=float)
        parser.add_argument('--Emax', default=10, type=float)
        parser.add_argument('--nbins', default=90000, type=int)
        parser.add_argument('--order', default=4)

    def init(self):
        ns = env.ns(self.opts.name)
        ns.reqparameter("Qp0", central=19968.9*500./10000000/1.04852, sigma=0)
        ns.reqparameter("Qp1", central=251.168*500./10000000/1.04852, sigma=0)
        ns.reqparameter("Qp2", central=-15.5711*500./10000000/1.04852, sigma=0)

        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1)
        orders = np.array([self.opts.order]*(len(edges)-1), dtype=int)

        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        with ns:
            model = ROOT.Quadratic()
            model2 = ROOT.Worst()

        model.QuaNL.old_bins(integrator.points.x)
        model2.WorstNL.old_bins(integrator.points.x)
        hist = ROOT.GaussLegendreHist(integrator)
        hist.hist.f(model.QuaNL.bins_after_nl)
        hist2 = ROOT.GaussLegendreHist(integrator)
        hist2.hist.f(model2.WorstNL.bins_after_nl)
        ns.addobservable('spectrum', hist.hist)
        ns.addobservable('spectrum2', hist2.hist)
