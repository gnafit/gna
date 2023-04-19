#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import pytest

from gna.unittest import allure_attach_file
from mpl_tools.helpers import savefig

from load import ROOT as R
from gna import constructors as C
import gna.constructors as C

from gna.env import env
from gna.parameters.oscillation import reqparameters

import itertools as it

#1 Unitarity-> sum_b P_ab - 1. < 1E-8
# assertLess(a, b) -> a < b
#2 The CPT theorem (P_ab -> P_ba, T-> -T)
#3 In the case of L = 0, P_ab = 0, P_aa = 1.
#4 Figure with (P_mue, P_amuae)

Enu = np.arange(700., 2000., step = 10)
amount_dots = len(Enu)

E_MeV = C.Points(Enu, labels='Neutrino energy')

ns = env.ns("matter_osc")
reqparameters(ns)
Delta = 0.
mass_ordering = 'normal'
ns['Delta'].set(Delta) # set CP-violating phase
ns['Alpha'].set(mass_ordering)
ns.defparameter("L", central = 810., fixed=True) # kilometers
ns.defparameter("rho", central = 2.75, fixed=True) # g/cm^3

f_nu = [R.Neutrino.e, R.Neutrino.mu, R.Neutrino.tau]
f_anu = [R.Neutrino.ae, R.Neutrino.amu, R.Neutrino.atau]

# Grouping flavor pairs for the first test.
all_flav_pairs = list(it.product(f_nu, f_nu)) + list(it.product(f_anu, f_anu))
triple_flav_pairs = [list(l) for _, l in it.groupby(all_flav_pairs, key=lambda x: x[0])]

def oscprob(namespace, init, final):
    with namespace:
        oscprob_matter = R.OscProbConstantDensity(init(), final(), labels='Oscillation probability in matter')
        E_MeV.points >> oscprob_matter.oscprob.Enu

    return oscprob_matter.oscprob.oscprob.data()

# The first test: sum_b P_ab = 1.
@pytest.mark.parametrize("flavors", triple_flav_pairs)
def test_unitarity(flavors):
    summ_oscprobs = np.zeros(amount_dots)
    for pair in flavors:
        initial_flavor, final_flavor = pair
        probs = oscprob(ns, initial_flavor, final_flavor)
        summ_oscprobs += probs
    assert np.allclose(summ_oscprobs, 1., rtol = 1E-8)

#(tmp_path):
#    import IPython
#    IPython.embed()
#    sys.exit()

# Grouping flavor pairs for the second test.
# List of all unique pairs without diagonal pairs.
uniq_flav_pairs = list(it.combinations(f_nu,2)) + list(it.combinations(f_anu,2))

# List of lists with unique pair in each one.
flav_pairs_ab = []
for pair in uniq_flav_pairs:
    flav_pairs_ab.append([pair])

# The second test: P_ab = P_ba.
@pytest.mark.parametrize("flavors", flav_pairs_ab)
def test_time_reversal(flavors):
    for pair in flavors:
        flavor_a, flavor_b = pair
        diff = oscprob(ns, flavor_a, flavor_b) - oscprob(ns, flavor_b, flavor_a)
    assert np.allclose(diff, 0., rtol = 1E-8)

ns_zero = env.ns("matter_osc_zero")
reqparameters(ns_zero)

ns_zero['Delta'].set(Delta) # set CP-violating phase
ns_zero['Alpha'].set(mass_ordering)
ns_zero.defparameter("L", central = 0., fixed=True) # kilometers
ns_zero.defparameter("rho", central = 2.75, fixed=True) # g/cm^3

# Diagonal flavor pairs. (For future work with the third test)
diag_flav_pairs = (
    list(
    set( it.combinations_with_replacement(f_nu,2) ) - set( it.combinations(f_nu,2) ) )
    + list(
    set( it.combinations_with_replacement(f_anu,2) ) - set( it.combinations(f_anu,2) ) )
    )

flav_pairs_0 = uniq_flav_pairs + diag_flav_pairs

# List of lists with unique pair (with diagonal pairs) in each one.
flav_pairs = []
for pair in flav_pairs_0:
    flav_pairs.append([pair])

# The third test: in case L = 0, P_ab = 0, P_aa = 1
@pytest.mark.parametrize("flavors", flav_pairs)
def test_zero_distance(flavors):
    for pair in flavors:
        initial_flavor, final_flavor = pair
        prob = oscprob(ns_zero, initial_flavor, final_flavor)

    if initial_flavor == final_flavor :
        x = 1.
    else:
        x = 0.

    assert np.allclose(prob, x, rtol = 1E-8)

def test_parameter_influence(tmp_path):
    _, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    osc_probs = {}
    modes = [(R.Neutrino.mu(), R.Neutrino.e()), (R.Neutrino.mu(), R.Neutrino.mu())]
    with ns:
        for mode in modes:
            oscprob_matter = R.OscProbConstantDensity(*mode, labels='Oscillation probability in matter')
            E_MeV.points >> oscprob_matter.oscprob.Enu
            osc_probs[mode] = oscprob_matter

    def plot_osc(ax):
        for oscprob, l in zip(osc_probs.values(), ['Electron appearance', 'Muon disappearance']):
            ax.plot(Enu, oscprob.oscprob.data(), label=l)
        ax.legend()
        ax.annotate(f'DeltaMSqEE = {ns["DeltaMSqEE"].value()}', xy=(1600, 0.6))

    plot_osc(axes[0])
    ns['DeltaMSqEE'].set(0.0028)
    plot_osc(axes[1])
    axes[1].set_xlabel('E, MeV')
    savefig(os.path.join(str(tmp_path), 'matter_oscprobs.png'))
