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
        points = np.array([1,2,3,4,5], dtype=float)
        pointset = ROOT.PointSet(points, len(points))

        points = np.array([1,2,3,4,5], dtype=float)
        pointset2 = ROOT.PointSet(points, len(points))

        pointset["points"].outputs[0].connect(oscprob["prob"].inputs[0])
        prediction.add(oscprob["prob"].outputs[0])
        prediction.add(pointset2["points"].outputs[0])
        print np.frombuffer(prediction.data(), dtype=float, count=10)
