#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle.detector_iav import detector_iav_from_file
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar

esmear, transf = detector_iav_from_file( 'output/iavMatrix_P14A_LS.root', 'iav_matrix', ndiag=4 )

import IPython
IPython.embed()
