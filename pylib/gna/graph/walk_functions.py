# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import ROOT as R

def taint(entry):
    if bool(entry.tainted):
        return
    entry.tainted.taint()

def taint_dummy(entry):
    if bool(entry.tainted):
        return
    bool(entry.tainted)

def size(sink):
    return sink.data.type.size()
