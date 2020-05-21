# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from gna.ui import basecmd
import ROOT
import gna.constructors as C
import gna.parameters.oscillation
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from pylab import figure, subplot, legend, plot, show, clf
from matplotlib import rc
rc('text', usetex=True)

class cmd(basecmd):
  def init(self):
    self.ns = self.env.ns("test_oscprob")
    gna.parameters.oscillation.reqparameters(self.ns)
    self.ns.defparameter("L", central=1.8, sigma=0)
    self.ns.defparameter("sigma", central=1.e-5, sigma=0)
    neutrinos = [ROOT.Neutrino.e(), ROOT.Neutrino.mu(), ROOT.Neutrino.tau()]
    L_arr = [2., 800., 12000.]
    Enu_arr = [np.linspace(1., 10., 100), np.linspace(1e2, 1e4, 10000)]
    self.make_prediction(ROOT.Neutrino.e(), ROOT.Neutrino.e(), Enu_arr[0], L_arr[0] )
    self.make_prediction(ROOT.Neutrino.mu(), ROOT.Neutrino.e(), Enu_arr[1], L_arr[1] )
    self.make_prediction(ROOT.Neutrino.mu(), ROOT.Neutrino.tau(), Enu_arr[1], L_arr[2] )
    self.make_simple_prediction(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(), Enu_arr[0], L_arr[0])

  def make_simple_prediction(self, from_nu, to_nu, Enu_arr, L):
    Enu_p = C.Points(Enu_arr)
    self.ns['L'].set(2.)
    with self.ns:
        oscprob_full = ROOT.OscProbPMNS(from_nu, to_nu)
        oscprob_full.full_osc_prob.inputs.Enu(Enu_p)
        data_osc = oscprob_full.full_osc_prob.oscprob
        data_osc.data()
        print(data_osc.data())
        plt.figure()
        plt.plot(Enu_arr, data_osc.data())
        plt.savefig('test.pdf')



  def make_prediction(self, from_neutrino, to_neutrino, Enu_arr, L):
    with self.ns:
      Enu = C.Points(Enu_arr)

      oscprob_classes = {
        'standard': ROOT.OscProbPMNS,
        'decoh': ROOT.OscProbPMNSDecoh,
      }
      oscprobs = {}
      data = {}
      for name, cls in oscprob_classes.iteritems():
          oscprob = cls(from_neutrino, to_neutrino)
          for compname in oscprob.probsum.inputs.keys():
            if compname != 'comp0':
              for tname, tf in oscprob.transformations.iteritems():
                if compname in tf.outputs:
                  oscprob.probsum[compname](tf[compname])
                  break
            else:
              oscprob.probsum[compname](C.Points(np.ones_like(Enu_arr)))
          for tf in oscprob.transformations.itervalues():
            if 'Enu' in tf.inputs:
              tf.inputs.Enu(Enu)
          self.ns.addobservable('probability_{0}'.format(name), oscprob.probsum)
          oscprobs[name] = oscprob
          data[name] = oscprob.probsum



      data_stand, data_decoh = data['standard'], data['decoh']

      self.ns["L"].set(L)
      self.ns["Delta"].set(0.0)
      sigma_arr = [1.e-17, 1e-1, 2e-1, 5e-1]
      nu_names = ['nu_e', 'nu_mu', 'nu_tau']
      filename = 'oscprob_'+nu_names[from_neutrino.flavor]+'_'+nu_names[to_neutrino.flavor]+'.pdf'
      self.open_pdf(filename, oscprobs['standard'].__class__.__name__)
      plt.plot(Enu_arr, data_stand.data(), label=r"$P_{PW}$", linewidth=3)
      for sigma in sigma_arr:
        self.ns["sigma"].set(sigma)
        plt.plot(Enu_arr, data_decoh.data(), label=r"$\sigma={0}$".format(sigma))
      #plt.show()
      nu_tex = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
      self.close_pdf('E, [MeV]', r'$P($'+nu_tex[from_neutrino.flavor]+r'$\to$'+nu_tex[to_neutrino.flavor]+r'$)$')

  def open_pdf(self, name, title):
    self.pp = PdfPages(name)
    plt.figure()
    plt.title(title)
    self.ax = subplot(111)

  def close_pdf(self, x_title, y_title):
    self.ax.set_xlabel(x_title, size='x-large')
    self.ax.set_ylabel(y_title, size='x-large')
    self.ax.set_xscale('log')
    plt.legend(loc=4).get_frame().set_alpha(0.6)


    plt.savefig(self.pp, format='pdf')
    self.pp.close()
