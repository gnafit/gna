# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R

def taint(entry):
    if bool(entry.tainted):
        return
    entry.tainted.taint()

def taint_dummy(entry):
    if bool(entry.tainted):
        return
    bool(entry.tainted)
