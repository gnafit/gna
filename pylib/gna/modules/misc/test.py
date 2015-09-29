from gna.ui import basecmd
import ROOT
import numpy as np

class test(basecmd):
    def run(self):
        env = self.env.default
        pars = env.pars
        pars.define('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
        pars.define('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                    limits=(0, 0.1))
        pars.define('L', central=52, sigma=0)

        oscprob = ROOT.OscProb2nu()
        prediction = ROOT.PredictionSet()
        edges = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3], dtype=float)
        orders = np.array([5]*(len(edges)-1), dtype=int)
        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        oscprob["prob"].inputs[0].connect(integrator["points"].outputs[0])
        integrator["hist"].inputs[0].connect(oscprob["prob"].outputs[0])
        prediction.add(integrator["hist"].outputs[0])
        print np.frombuffer(prediction.data(), dtype=float, count=4)
        pars['SinSq12'].set(0)
        print np.frombuffer(prediction.data(), dtype=float, count=4)
