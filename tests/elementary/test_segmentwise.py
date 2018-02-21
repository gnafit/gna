#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from constructors import SegmentWise, Points
from gna.bindings import DataType

segments = N.arange(0.0, 10.0, dtype='d')
points   = N.arange(-0.5, 11.0, dtype='d')
points_t = Points( points )

sw = SegmentWise(segments)
sw.segments.points( points_t.points.points )

diff = segments - sw.segments.edges.data()
if not (N.fabs(diff)<1.e-16).all():
    print( '\033[31mFailed to get the edges back\033[0m' )
    print(diff)

