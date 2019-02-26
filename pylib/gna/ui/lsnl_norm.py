from gna.ui import basecmd
from gna.env import env
import ROOT
import numpy as np
import array
import constructors as C

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--Emin', default=1.0, type=float)
        parser.add_argument('--Emax', default=10, type=float)
        parser.add_argument('--nbins', default=900, type=int)
        parser.add_argument('--order', default=4)

    def init(self):
        ns = env.ns(self.opts.name)
        ns.reqparameter("Qp0", central=0.0065, sigma=0)
        ns.reqparameter("Qp1", central=0, sigma=0)
        ns.reqparameter("Qp2", central=1391, sigma=0)
        ns.reqparameter("Qp3", central=1.0, sigma=0)

        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1)
        orders = np.array([self.opts.order]*(len(edges)-1), dtype=int)

        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        with ns:
            model2 = ROOT.Mine()
            model3 = ROOT.Mine()

        model2.DisplayNL.old_bins2(integrator.points.x)
        model3.DisplayNL.old_bins2(integrator.points.x)
        normfactor=2
        print('normfactor: {0}'.format(normfactor))
        model3.setnorm(normfactor)
        model3.normMineNL.new_bins(model3.DisplayNL.bins2_after_nl)
        hist2 = ROOT.GaussLegendreHist(integrator)
        hist2.hist.f(model2.DisplayNL.bins2_after_nl)
        hist3 = ROOT.GaussLegendreHist(integrator)
        hist3.hist.f(model3.normMineNL.norm_new_bins)
        ns.addobservable('spectrum2', hist2.hist)
        ns.addobservable('spectrum3', hist3.hist)
