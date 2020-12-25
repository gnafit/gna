#!/usr/bin/env python

from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.env import env
from mpl_tools.helpers import savefig, plot_hist, add_colorbar
from gna.converters import convert
from argparse import ArgumentParser
import gna.constructors as C

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-n', '--njoin', type=int, default=2)
args = parser.parse_args()

edges   = N.linspace(0, 10, 17, dtype='d')

ntrue = C.Histogram(edges, N.ones(edges.size-1) )
rebin = R.RebinN(args.njoin)
ntrue >> rebin

olddata = ntrue.single().data()
newdata = rebin.single().data()
olddatatype = ntrue.single().datatype()
newdatatype = rebin.single().datatype()

print('Old data:', olddata, olddatatype)
print('New data:', newdata, newdatatype)
