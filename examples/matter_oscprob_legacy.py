#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np 
import argparse
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True #prevents ROOT from hijacking sys.argv

from gna.env import env
from gna.parameters.oscillation import reqparameters
from gna.bindings import common
import gna.constructors as C

_flavors = { "e":   ROOT.Neutrino.e(), 
             "mu":  ROOT.Neutrino.mu(),
             "tau": ROOT.Neutrino.tau(),
             "ae":  ROOT.Neutrino.ae(),
             "amu": ROOT.Neutrino.amu(),
             "atu": ROOT.Neutrino.atau()
             }

def example():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--energy", default=[700., 2000., 10], nargs=3, type=float, help='Set neutrino energy range and step in MeV')
    parser.add_argument('-f', '--flavors', default=['mu', 'mu'], nargs=2, help='Set initital and final flavors for oscillation probability')
    parser.add_argument('-r', '--density', default=2.75, type=float, help='Set density of matter in g/cm^3')
    parser.add_argument('-L', '--distance', default=810., type=float, help='Set distance of propagation')
    parser.add_argument('--mass-ordering', default='normal', choices=['normal', 'inverted'], help='Set neutrino mass ordering (hierarchy)')
    parser.add_argument('--print-pars', action='store_true', help='Print all parameters in namespace')
    opts = parser.parse_args()
    # Parsed arguments are put into opts object (of type argparse.Namespace) as attributes and can be accessed with a '.'
    # like in examples below. 
    # Note that argparse translate dashes '-' into underscores '_' for attribute names.

    #initialize energy range
    Enu = np.arange(opts.energy[0], opts.energy[1], step=opts.energy[2])
    E_MeV = C.Points(Enu, labels='Neutrino energy')

    # initialize initial and final flavors for neutrino after propagating in media
    try:
        initial_flavor, final_flavor = tuple(_flavors[flav] for flav in opts.flavors)
    except KeyError:
        raise KeyError("Invalid flavor is requested: try something from e, mu, tau")

    ns = env.ns("matter_osc")
    reqparameters(ns) # initialize oscillation parameters in namespace 'matter_osc'

    ns['Delta'].set(0) # set CP-violating phase to zero
    ns['SinSq23'].set(0.6) # change value of sin²θ₂₃, it will cause recomputation of PMNS matrix elements that depend of it
    ns['Alpha'].set(opts.mass_ordering) # Choose mass ordering
    ns.defparameter("L", central=opts.distance, fixed=True) # kilometres
    ns.defparameter("rho",central=opts.density, fixed=True) # g/cm^3

    # Print values of all parameters in namespace if asked
    if opts.print_pars:
        ns.printparameters()

    #initialize neutrino oscillation probability in matter.
    # All neccessary parameters such as values of mixing angles, mass splittings,
    # propagation distance and density of matter are looked up in namespace ns
    # upon creation of ROOT.OscProbMatter. 
    with ns:
        oscprob_m = ROOT.OscProbMatter(initial_flavor, final_flavor, labels='Oscillation probability in matter')
        # Add output of plain array of energies as input "Enu" to oscillation
        # probability. It means that oscillation probabilities would be computed
        # for each energy in array and array of it would be
        # provided as output
        E_MeV.points >> oscprob_m.oscprob.Enu

    # Accessing the outputs of oscillation probability for plotting

    osc_prob = oscprob_m.oscprob.oscprob.data()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.plot(Enu, osc_prob)

    ax.set_title("Oscillation probability in matter, {0} -> {1}".format(opts.flavors[0], opts.flavors[1]), fontsize=16)
    ax.set_xlabel("Neutrino energy, MeV", fontsize=14)
    ax.set_ylabel("Oscprob", fontsize=14)
    ax.grid(alpha=0.5)

    plt.show()

if __name__ == '__main__':
    example()
