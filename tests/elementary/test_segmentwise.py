#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from constructors import SegmentWise

e = N.arange(10, dtype='d')
sw = SegmentWise(e)

e1 = sw.segments.edges.data()

diff = e-e1
print( diff )

import IPython
IPython.embed()
