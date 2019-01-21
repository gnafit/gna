#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ROOT
from gna.env import env
from physlib import pdg
curpdg = pdg[2016]


# TODO: Add the way to automaticaly detect the current mass ordering and
# switch the values of mixings to the current known global best fit for a
# given ordering
# For a given moment all parameters are just set for normal ordering

# DeltaMSqIJ = m^2_j - m^2_i

def reqparameters(ns):
    ns.reqparameter('SinSq12', central=curpdg['sinSqtheta12'],
                      sigma=curpdg['sinSqtheta12_e'], limits=(0,1), label='Solar mixing angle sin²θ₁₂')

    ns.reqparameter('DeltaMSq12',central=curpdg['dmSq21'],
                      sigma=curpdg['dmSq21_e'], limits=(0, 0.1), label='Solar mass splitting Δm²₂₁')

    ns.reqparameter('SinSq13', central=curpdg['sinSqtheta13'],
                      sigma=curpdg['sinSqtheta13_e'], limits=(0,1), label='Reactor mixing angle sin²θ₁₃ ')

    ns.reqparameter('Alpha', type='discrete', default='normal',
            variants={'normal': 1.0, 'inverted': -1.0}, label='Neutrino mass hierarchy')

    ns.reqparameter('SinSq23', central=curpdg['sinSqtheta23_normal'],
                      sigma=curpdg['sinSqtheta23_normal_e'], limits=(0,1), label='Atmospheric mixing angle sin²θ₂₃')

    #  ns.reqparameter('DeltaMSq23', central=curpdg['dmSq32_normal'],
                     #  sigma=curpdg['dmSq32_normal_e'], limits=(0, 0.1))

    ns.reqparameter('DeltaMSqEE', central=curpdg['dmSqEE'],
                      sigma=curpdg['dmSqEE_e'], limits=(0, 0.1), label='Reactor average mass splitting Δm²(ee)')
    ns.reqparameter('Delta', type='uniformangle', central=0.0, label='CP violation phase δ(CP)')
    ns.reqparameter("SigmaDecohRel", central=1.e-5, sigma=1e-5, label='Relative momentum spread (decoherence)')

    with ns:
        ROOT.OscillationExpressions(ns=ns)
        ROOT.PMNSExpressions(ns=ns)

    ns['DeltaMSq23'].setLabel('Mass splitting (2, 3)')
    ns['DeltaMSq13'].setLabel('Mass splitting (1, 3)')
    ns['DeltaMSqMM'].setLabel('Accelerator average mass splitting Δm²(μμ)')
    ns['Phase'].setLabel('CP violation factor exp(-iδ)')
    ns['PhaseC'].setLabel('Conjugated CP violation factor exp(+iδ)')

    flavors = [ 'e', 'μ', 'τ' ]
    with ns:
        for iflavor in range(3):
            for jmass in range(3):
                name = 'V{0}{1}'.format(iflavor,jmass)
                ns[name].get()
                ns[name].setLabel('PMNS element {0}{1}'.format(flavors[iflavor], jmass))
