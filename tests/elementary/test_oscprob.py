#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import sys
import numpy as np
import matplotlib.pyplot as plt
import gna.parameters.oscillation
from gna import constructors as C # construct objects
from gna.bindings import common   # plot outputs
from gna.graphviz import savegraph
from mpl_tools.helpers import savefig
from gna.env import env
from gna import context, bindings

#
# Define the parameters
#
def test_oscprob():
    baselinename='L'
    ns = env.ns("testoscprob")
    gna.parameters.oscillation.reqparameters(ns)
    ns.defparameter(baselinename, central=2.0, fixed=True, label='Baseline, km')

    # Define energy range
    enu_input = np.arange(1.0, 10.0, 0.01)
    enu = C.Points(enu_input, labels='Neutrino energy, MeV')

    # Initialize oscillation variables
    component_names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
    with ns:
        R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), component_names, ns=ns)

    # Initialize neutrino oscillations
    with ns:
        labels=['Oscillation probability|%s'%s for s in ('component 12', 'component 13', 'component 23', 'full', 'probsum')]
        oscprob = C.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae(), baselinename, labels=labels)

    enu >> oscprob.full_osc_prob.Enu
    enu >> (oscprob.comp12.Enu, oscprob.comp13.Enu, oscprob.comp23.Enu)

    # Oscillation probability as single transformation
    op_full = oscprob.full_osc_prob.oscprob

    # Oscillation probability as weighted sum
    unity = C.FillLike(1, labels='Unity')
    enu >> unity.fill.inputs[0]
    with ns:
        op_sum = C.WeightedSum(component_names, [unity.fill.outputs[0], oscprob.comp12.comp12, oscprob.comp13.comp13, oscprob.comp23.comp23], labels='Oscillation probability sum')

    # Print some information
    oscprob.print()
    print()

    ns.printparameters(labels=True)

    # Print extended information
    oscprob.print(data=True, slice=slice(None, 5))

    assert np.allclose(op_full.data(), op_sum.data())

    if "pytest" not in sys.modules:
        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( 'E nu, MeV' )
        ax.set_ylabel( 'P' )
        ax.set_title( 'Oscillation probability' )

        op_full.plot_vs(enu.single(), '-', label='full oscprob')
        op_sum.plot_vs(enu.single(), '--', label='oscprob (sum)')

        ax.legend(loc='lower right')

        savefig('output/test_oscprob.pdf')
        savegraph(enu, 'output/test_oscprob_graph.dot', namespace=ns)
        savegraph(enu, 'output/test_oscprob_graph.pdf', namespace=ns)

        plt.show()

if __name__ == "__main__":
    test_oscprob()
