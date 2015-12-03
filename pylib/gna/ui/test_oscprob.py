from gna.ui import basecmd
import ROOT
import gna.parameters.oscillation
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from pylab import figure, subplot, legend, plot, show, clf


class cmd(basecmd):
  def init(self):
    ns = self.env.ns("test")
    gna.parameters.oscillation.defparameters(ns)
    ns.defparameter("L", central=2, sigma=0)
    ns.defparameter("sigma", central=1.e-5, sigma=0)
    with ns:
      oscprob = ROOT.OscProbPMNSDecoh(ROOT.Neutrino.e(), ROOT.Neutrino.e())
      Enu_arr = np.linspace(1e-1, 10, 10000)
      Enu = ROOT.Points(Enu_arr)
      comp0 = ROOT.Points(np.ones_like(Enu_arr))
      oscprob.probsum['comp0'](comp0)
      for compname in (x for x in oscprob.transformations if x.startswith('comp')):
        oscprob[compname].inputs(Enu)
        oscprob.probsum[compname](oscprob[compname])
      ns.addobservable('probability',oscprob.probsum)
      prediction = ROOT.Prediction()
      prediction.append(oscprob.probsum)
      data = np.frombuffer(prediction.data(), count=prediction.size())
      print data
      x = ns["L"].value()/Enu_arr
      sigma_arr = [1.e-17,1e-1, 2e-1,5e-1]
      self.open_pdf('test.pdf',oscprob.__class__.__name__)
      for sigma in sigma_arr:
        ns["sigma"].set(sigma)
        prediction.update()
        self.make_plot(list(reversed(x)),list(reversed(data)),r"$\sigma={0}$".format(sigma))
      #plt.show()
      self.close_pdf()
      
  def open_pdf(self,name,title):
    self.pp = PdfPages(name)
    plt.figure()
    plt.title(title)
    self.ax = subplot(111)
    
  def close_pdf(self):
    plt.legend(loc=3)
    plt.savefig(self.pp,format='pdf')
    self.pp.close()
    
  def make_plot(self,x,y,label):
    plt.plot(x, y, label=label)
    self.ax.set_xlabel('L/E', size='x-large')
    self.ax.set_ylabel('Probability', size='x-large')
  