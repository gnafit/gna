from gna.ui import basecmd
import ROOT
import gna.parameters.oscillation
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from pylab import figure, subplot, legend, plot, show, clf


class cmd(basecmd):
  def init(self):
    self.ns = self.env.ns("test_oscprob")
    gna.parameters.oscillation.defparameters(self.ns)
    self.ns.defparameter("L", central=2, sigma=0)
    self.ns.defparameter("sigma", central=1.e-5, sigma=0)
    neutrinos = [ROOT.Neutrino.e(), ROOT.Neutrino.mu(), ROOT.Neutrino.tau()]
    L_arr = [2,800,12000]
    Enu_arr = [np.linspace(1e-1, 10, 10000), np.linspace(1e2,1e4,10000)]
    self.make_prediction(ROOT.Neutrino.e(), ROOT.Neutrino.e(), Enu_arr[0], L_arr[0] )
    self.make_prediction(ROOT.Neutrino.mu(), ROOT.Neutrino.e(), Enu_arr[1], L_arr[1] )
    self.make_prediction(ROOT.Neutrino.mu(), ROOT.Neutrino.tau(), Enu_arr[1], L_arr[2] )

    
    
  def make_prediction(self,from_neutrino, to_neutrino, Enu_arr, L):
    with self.ns:
      Enu = ROOT.Points(Enu_arr)
      
      oscprob_classes = {
        'standard': ROOT.OscProbPMNS,
        'decoh': ROOT.OscProbPMNSDecoh,
      }
      oscprobs = {}
      predictions = {}
      data = {}
      for name, cls in oscprob_classes.iteritems():
          oscprob = cls(from_neutrino, to_neutrino)
          try:
            comp0 = ROOT.Points(np.ones_like(Enu_arr))
            oscprob.probsum['comp0'](comp0)
          except AttributeError:
            pass
          for compname in (x for x in oscprob.transformations if x.startswith('comp')):
            oscprob[compname].inputs(Enu)
            oscprob.probsum[compname](oscprob[compname])
          self.ns.addobservable('probability_{0}'.format(name),oscprob.probsum)
          oscprobs[name] = oscprob
          predictions[name] = ROOT.Prediction()
          predictions[name].append(oscprob.probsum)
          data[name] = np.frombuffer(predictions[name].data(), count=predictions[name].size())

      data_stand, data_decoh = data['standard'], data['decoh']

      self.ns["L"].set(L)
      self.ns["Delta"].set(0.0)
      sigma_arr = [1.e-17,1e-1,2e-1,5e-1]
      nu_names = ['nu_e', 'nu_mu', 'nu_tau']
      filename = 'oscprob_'+nu_names[from_neutrino.flavor]+'_'+nu_names[to_neutrino.flavor]+'.pdf'
      self.open_pdf(filename,oscprobs['standard'].__class__.__name__)
      predictions['standard'].update()
      plt.plot(Enu_arr,data_stand,label=r"$P_{PW}$", linewidth=3)
      for sigma in sigma_arr:
        self.ns["sigma"].set(sigma)
        predictions['decoh'].update()
        plt.plot(Enu_arr,data_decoh,label=r"$\sigma={0}$".format(sigma))
      #plt.show()
      nu_tex = [r"$\nu_e$",r"$\nu_\mu$",r"$\nu_\tau$"]
      self.close_pdf('E, [MeV]',r'$P($'+nu_tex[from_neutrino.flavor]+r'$\to$'+nu_tex[to_neutrino.flavor]+r'$)$')
  
  def open_pdf(self,name,title):
    self.pp = PdfPages(name)
    plt.figure()
    plt.title(title)
    self.ax = subplot(111)
    
  def close_pdf(self,x_title,y_title):
    self.ax.set_xlabel(x_title, size='x-large')
    self.ax.set_ylabel(y_title, size='x-large')
    plt.legend(loc=3)
    plt.savefig(self.pp,format='pdf')
    self.pp.close()
    