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
        parser.add_argument('--Emin', default=1.2, type=float)
        parser.add_argument('--Emax', default=10, type=float)
        parser.add_argument('--nbins', default=8800, type=int)
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
            model1 = ROOT.Quadratic()
            model2 = ROOT.Worst()
            model3 = ROOT.Mine()

        model1.QuaNL.old_bins(integrator.points.x)
        model2.WorstNL.old_bins(integrator.points.x)
        model3.DisplayNL.old_bins2(integrator.points.x)
        itemindex = np.where(integrator.points.x.data()<2.28766)
        thisindex=len(itemindex[0])
        #print(mineedges.data()[thisindex-1])
        normfactor=1000.0/model3.DisplayNL.bins2_after_nl.data()[thisindex-1]
        #print('thisindex: {1} normfactor: {0}'.format(normfactor,thisindex))
        model3.setnorm(normfactor)
        model3.normMineNL.new_bins(model3.DisplayNL.bins2_after_nl)
        hist1 = ROOT.GaussLegendreHist(integrator)
        hist1.hist.f(model1.QuaNL.bins_after_nl)
        hist2 = ROOT.GaussLegendreHist(integrator)
        hist2.hist.f(model2.WorstNL.bins_after_nl)
        hist3 = ROOT.GaussLegendreHist(integrator)
        hist3.hist.f(model3.normMineNL.norm_new_bins)
        #print(integrator.points.x.data())
        #print(model3.normMineNL.norm_new_bins.data())
        #hist3.hist.f(model3.MineNL.bins_after_nl)
        #print(model1.QuaNL.bins_after_nl)
        #print(model2.WorstNL.bins_after_nl)
        #print(model3.MineNL.bins_after_nl)
        #ns.addobservable('spectrum1', hist1.hist)
        ns.addobservable('spectrum2', hist2.hist)
        ns.addobservable('spectrum3', hist3.hist)
        fdyb = ROOT.TFile.Open("data/dayabay/tmp/detector_nl_consModel_450itr.root")
        grdyb = fdyb.Get("positronScintNL")
        x1buff = grdyb.GetX()
        y1buff = grdyb.GetY()
        x1buff.SetSize(grdyb.GetN())
        y1buff.SetSize(grdyb.GetN())
        X2 = array.array('d',x1buff)
        Y2 = array.array('d',y1buff)
        xx2 = np.array([X2],dtype=float)
        xx2 = xx2[0][:]-np.array([(xx2[0][1]-xx2[0][0])/2])
        #print('before {0}'.format(xx2.shape))
        yy2 = np.array([Y2],dtype=float)
        yy2 = yy2[0][:]-np.array([0])
        histdyb = C.Histogram(xx2[:], yy2[:-1])
        ns.addobservable('dyb', histdyb.hist)

        fmore = ROOT.TFile.Open("compareNL_gr.root")
        grmore = fmore.Get("grmore")
        x1morebuff = grmore.GetX()
        y1morebuff = grmore.GetY()
        x1morebuff.SetSize(grmore.GetN())
        y1morebuff.SetSize(grmore.GetN())
        X2more = array.array('d',x1morebuff)
        Y2more = array.array('d',y1morebuff)
        xx2more = np.array([X2more],dtype=float)
        xx2more = xx2more[0][:]-np.array([(xx2more[0][1]-xx2more[0][0])/2])
        #print('before {0}'.format(xx2more.shape))
        yy2more = np.array([Y2more],dtype=float)
        yy2more = yy2more[0][:]-np.array([0])
        histmore = C.Histogram(xx2more[:], yy2more[:-1])
        ns.addobservable('grmore', histmore.hist)

        fless = ROOT.TFile.Open("compareNL_gr.root")
        grless = fless.Get("grless")
        x1lessbuff = grless.GetX()
        y1lessbuff = grless.GetY()
        x1lessbuff.SetSize(grless.GetN())
        y1lessbuff.SetSize(grless.GetN())
        X2less = array.array('d',x1lessbuff)
        Y2less = array.array('d',y1lessbuff)
        xx2less = np.array([X2less],dtype=float)
        xx2less = xx2less[0][:]-np.array([(xx2less[0][1]-xx2less[0][0])/2])
        #print('before {0}'.format(xx2less.shape))
        yy2less = np.array([Y2less],dtype=float)
        yy2less = yy2less[0][:]-np.array([0])
        histless = C.Histogram(xx2less[:], yy2less[:-1])
        ns.addobservable('grless', histless.hist)
