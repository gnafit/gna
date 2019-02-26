from gna.ui import basecmd
from matplotlib import pyplot as P
from gna.env import env
import ROOT
import numpy as np
import array
import constructors as C
from converters import convert
from mpl_tools.helpers import add_colorbar, plot_hist, savefig

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--Emin', default=1.0, type=float)
        parser.add_argument('--Emax', default=10, type=float)
        parser.add_argument('--nbins', default=9, type=int)
        parser.add_argument('--order', default=4)

    def init(self):
        ns = env.ns(self.opts.name)
        ns.reqparameter("Qp0", central=0, sigma=0) #0.0065
        ns.reqparameter("Qp1", central=0, sigma=0)
        ns.reqparameter("Qp2", central=1391, sigma=0)
        ns.reqparameter("Qp3", central=0.0, sigma=0)

        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1)
        orders = np.array([self.opts.order]*(len(edges)-1), dtype=int)

        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        with ns:
            model2 = ROOT.Mine()
            model3 = ROOT.Mine()
        trypoints = C.Points(edges)
        model2.MineNL.old_bins(trypoints)
        model3.MineNL.old_bins(trypoints)
        worstedges2 = model2.MineNL.bins_after_nl
        lsnl_model_2 = ROOT.HistNonlinearity(True)
        lsnl_model_2.set( trypoints, worstedges2)


        #mat2 = convert(lsnl_model_2.getDenseMatrix(), 'matrix')
        #mat2 = np.ma.array( mat2, mask= mat2==0.0 )
        mat2 = lsnl_model_2.matrix.FakeMatrix.data()
        print( mat2.sum( axis=0 ) )

        fig = P.figure()
        ax_mat = P.subplot( 111 )
        c = ax_mat.matshow( mat2, cmap="viridis")
        add_colorbar( c )
        P.show()


        normfactor=1.1
        print('normfactor: {0}'.format(normfactor))
        model3.setnorm(normfactor)
        model3.normMineNL.new_bins(model3.MineNL.bins_after_nl)
        worstedges3 = model3.normMineNL.norm_new_bins
        lsnl_model_3 = ROOT.HistNonlinearity(True)
        lsnl_model_3.set( trypoints, worstedges3)


        mat3 = lsnl_model_3.matrix.FakeMatrix.data()
        print( mat3.sum( axis=0 ) )

        fig3 = P.figure()
        ax_mat3 = P.subplot( 111 )
        c3 = ax_mat3.matshow( mat3, cmap="viridis")
        add_colorbar( c3 )
        P.show()
