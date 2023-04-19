#!/usr/bin/env python

from load import ROOT as R
from gna import constructors as C
from gna.env import env
from gna.unittest import *
from matplotlib import pyplot as plt
import numpy as np
import os

def test_energyresolutionsigmarelabc_v01(tmp_path):
    #
    # Define the parameters in the current namespace
    #
    ns = env.globalns('test_energyresolutionsigmarelabc_v01')
    weights = [ 'Eres_'+s for s in 'abc' ]
    wvals   = [0.016, 0.081, 0.026]
    percent = 0.01
    ns.defparameter(weights[0],  central=wvals[0], fixed=True)
    par = ns.defparameter(weights[1],  central=wvals[1], fixed=True)
    ns.defparameter(weights[2],  central=wvals[2], fixed=True)
    ns.printparameters()

    values = []
    def pop_value():
        values, par
        par.set(values.pop())

    def push_value(v):
        values, par
        values.append(par.value())
        par.set(v)

    #
    # Define bin edges
    #
    Energy = np.linspace(1.0, 8.0, 200)
    energy = C.Points(Energy)

    with ns:
        sigma = C.EnergyResolutionSigmaRelABC(weights)

    energy >> sigma

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='Energy', ylabel=r'$\sigma/E$', title='')
    ax.minorticks_on()
    ax.grid()

    sigma.sigma.sigma.plot_vs(energy.points.points, '-')
    path = os.path.join(str(tmp_path), 'eres_sigma_rel.png')
    plt.savefig(path)
    allure_attach_file(path)

    path = os.path.join(str(tmp_path), 'eres_sigma_rel_graph.png')
    savegraph(sigma.sigma, path)
    allure_attach_file(path)

    cmp = sigma.sigma.sigma.data()
    cmpto = (wvals[0]**2 + wvals[1]**2/Energy + (wvals[2]/Energy)**2)**0.5
    assert np.allclose(cmp, cmpto, rtol=1.e9, atol=1.e-14)

