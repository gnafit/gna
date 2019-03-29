#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gna.bindings import patchROOTTemplate, ROOT as R
import numpy as N

@patchROOTTemplate( (R.arrayview, R.arrayviewAllocator), 'view' )
def arrayview__view(self):
    buf = self.data()
    return N.frombuffer(buf, count=self.size(), dtype=buf.typecode)

@patchROOTTemplate( (R.arrayviewAllocator), 'viewall' )
def arrayviewAllocator__viewall(self):
    buf = self.data()
    return N.frombuffer(buf, count=self.maxSize(), dtype=buf.typecode)

