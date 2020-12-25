#!/usr/bin/env python

from load import ROOT as R
from matplotlib import pyplot as P
import numpy as N
from gna.env import env
from gna.labelfmt import formatter as L
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from scipy.stats import norm
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C

P.rc('text', usetex=True)

parser = ArgumentParser()
parser.add_argument( '-o', '--output' )
parser.add_argument( '-s', '--show', action='store_true' )
opts = parser.parse_args()

env.defparameter("E_0", central=0.165, fixed=True)
env.defparameter("p0",  central=-7.26624e+00, fixed=True)
env.defparameter("p1",  central=1.72463e+01,  fixed=True)
env.defparameter("p2",  central=-2.18044e+01, fixed=True)
env.defparameter("p3",  central=1.44731e+01,  fixed=True)
env.defparameter("p4",  central=3.22121e-02,  fixed=True)

energy = N.linspace(0.1, 10, num=200)
gna_energy = C.Points(energy)
cherenkov = R.Cherenkov_Borexino()
cherenkov.cherenkov.energy(gna_energy)
ch_response = cherenkov.cherenkov.ch_npe.data()

fig = P.figure()
ax = P.subplot( 111 )
ax.plot(energy, ch_response)
ax.set_yscale("log")
ax.set_xlabel(r"E$_{e}$, MeV", fontsize=14)
ax.set_ylabel(r"Number of p.e.", fontsize=14)
ax.set_title("Cherenkov radiation contribution", fontsize=18)
ax.grid(alpha=0.5)
# ax.set_xscale("log")

if opts.show:
    P.show()

if opts.output:
    P.savefig(opts.output)

#  import IPython; IPython.embed()
